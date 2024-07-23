import numpy as np
import pygame
from dataclasses import dataclass

from skimage.draw import random_shapes
from src.gym.grid import GridGym, GridRewardParams, GridRenderParams, GridStateRenderParams
from src.gym.utils import get_visibility_map, draw_text, draw_shape_matrix, is_solvable
from seaborn import color_palette


class RandomTargetGenerator:
    @dataclass
    class Params:
        coverage_range: (float, float) = (0.2, 0.5)
        shape_range: (int, int) = (3, 8)

    def __init__(self, params: Params):
        self.params = params

    def generate_target(self, obstacles):

        area = np.product(obstacles.shape)

        target = self.__generate_random_shapes_area(
            self.params.shape_range[0],
            self.params.shape_range[1],
            area * self.params.coverage_range[0],
            area * self.params.coverage_range[1],
            obstacles.shape
        )

        return target & ~obstacles

    def __generate_random_shapes(self, min_shapes, max_shapes, shape):
        img, _ = random_shapes(shape, max_shapes, min_shapes=min_shapes, channel_axis=None,
                               allow_overlap=True, rng=np.random.randint(2 ** 32 - 1))
        # Numpy random usage for random seed unifies random seed which can be set for repeatability
        attempt = np.array(img != 255, dtype=bool)
        return attempt, np.sum(attempt)

    def __generate_random_shapes_area(self, min_shapes, max_shapes, min_area, max_area, shape, retry=100):
        for attemptno in range(retry):
            attempt, area = self.__generate_random_shapes(min_shapes, max_shapes, shape)
            if min_area is not None and min_area > area:
                continue
            if max_area is not None and max_area < area:
                continue
            return attempt
        print("Was not able to generate shapes with given area constraint in allowed number of tries."
              " Randomly returning next attempt.")
        attempt, area = self.__generate_random_shapes(min_shapes, max_shapes, shape)
        print("Size is: ", area)
        return attempt

    def __generate_exclusive_shapes(self, exclusion, min_shapes, max_shapes, shape):
        attempt, area = self.__generate_random_shapes(min_shapes, max_shapes, shape)
        attempt = attempt & (~exclusion)
        area = np.sum(attempt)
        return attempt, area

    # Create target image and then subtract exclusion area
    def __generate_exclusive_shapes_area(self, exclusion, min_shapes, max_shapes, min_area, max_area, shape, retry=100):
        for attemptno in range(retry):
            attempt, area = self.__generate_exclusive_shapes(exclusion, min_shapes, max_shapes, shape)
            if min_area is not None and min_area > area:
                continue
            if max_area is not None and max_area < area:
                continue
            return attempt

        print("Was not able to generate shapes with given area constraint in allowed number of tries."
              " Randomly returning next attempt.")
        attempt, area = self.__generate_exclusive_shapes(exclusion, min_shapes, max_shapes, shape)
        print("Size is: ", area)
        return attempt


class SimpleSquareCamera:

    def __init__(self, camera_half_length: int, shadowing: np.ndarray):
        chl = camera_half_length
        self.camera_half_length = chl
        self.size = np.array([2 * chl + 1] * 2)
        self.visibility_map = get_visibility_map(camera_half_length, shadowing_map=shadowing)

    def compute_view(self, position):
        x_pos, y_pos = position
        return self.visibility_map[x_pos, y_pos]


@dataclass
class CPPRewardParams(GridRewardParams):
    cell_reward: float = 0.4
    completion_reward: float = 0.0


@dataclass
class CPPStateRenderParams(GridStateRenderParams):
    draw_decomp: bool = False
    draw_view: bool = False


@dataclass
class CPPRenderParams(GridRenderParams):
    normal_state: CPPStateRenderParams = CPPStateRenderParams()
    terminal_state: CPPStateRenderParams = CPPStateRenderParams()


class CPPGym(GridGym):
    @dataclass
    class Params(GridGym.Params):
        target_generator: RandomTargetGenerator.Params = RandomTargetGenerator.Params()
        camera_half_length: int = 2

        rewards: CPPRewardParams = CPPRewardParams()
        rendering: CPPRenderParams = CPPRenderParams()

    @dataclass
    class Init(GridGym.Init):
        target: np.ndarray

    @dataclass
    class State(GridGym.State):
        coverage = None
        cells_remaining = 0
        decomp = []
        decomp_init = None

    def __init__(self, params: Params = Params()):
        super().__init__(params)
        self.generator = RandomTargetGenerator(params.target_generator)

        self.params = params
        self._camera = []
        for map_image in self._map_image:
            self._camera.append(SimpleSquareCamera(params.camera_half_length, map_image.shadowing()))

        for k, (m, c) in enumerate(zip(self._map_image, self._camera)):
            if not is_solvable(m.landing_distances(), m.obst, c.visibility_map, self.max_budget):
                print(f"{self.map_names[k]} is not solvable.")
                exit(1)

        self.layer_order = ["environment", "decomp", "trajectory", "view", "agent", "decomp_labels"]

    @property
    def padding_values(self):
        return [0, 1, 1, 0, 0]

    def create_state(self):
        return CPPGym.State()

    def initialize_map(self, state, map_index=None):
        self._initialize_map(state, map_index)
        target = np.zeros(state.map.shape[:2] + (1,), dtype=bool)
        state.map = np.concatenate((state.map, target), axis=-1)
        state.coverage = np.zeros(state.map.shape[:2], dtype=bool)

    def add_map(self, filename):
        index = super().add_map(filename)
        camera = SimpleSquareCamera(self.params.camera_half_length, self._map_image[index].shadowing())
        if index == len(self._camera):
            self._camera.append(camera)
        else:
            self._camera.insert(index, camera)
        if is_solvable(self._map_image[index].landing_distances(), self._map_image[index].obst,
                       self._camera[index].visibility_map, self.params.budget_range[1]):
            return index
        return -1

    def reset(self, state=None, seed=None, options=None, init=None, map_name=None):
        if state is None:
            state = self.state

        if init is None:
            init = self.generate_init(map_name)
        state.init = init
        self.initialize_map(state, self.get_map_index(init.map_name))
        state.map[..., -1] = init.target
        state.decomp_init = state.map[..., 3].copy()
        state.decomp = []
        self.reset_state(state, init)
        state.cells_remaining = self.get_remaining_cells(state)

        obs = self.get_obs(state)

        self.render(state)

        return obs, self.get_info(state)

    def generate_init(self, map_name=None) -> Init:
        init = super().generate_init(map_name)
        map_index = self.get_map_index(init.map_name)
        shape = self._map_image[map_index].original_shape
        self.generator.shape = shape
        cropped_target = self.generator.generate_target(self._map_image[map_index].obst)
        target = np.zeros(self.shape)
        target[:shape[0], :shape[1]] = cropped_target
        return CPPGym.Init(position=init.position, budget=init.budget, map_name=init.map_name, target=target)

    def step(self, action, state=None):
        if state is None:
            state = self.state
        rewards = self._step(state, action)

        obs = self.get_obs(state)

        self.render(state)

        return obs, rewards, state.terminated, state.truncated, self.get_info(state)

    def _step(self, state, action):
        if isinstance(action, np.ndarray):
            action = action[0]
        self.motion_step(state, action)
        self.vision_step(state)
        rewards = self.get_rewards(state)
        state.episodic_reward += rewards
        state.truncated = state.steps >= self.timeout_steps(state)
        return rewards

    def vision_step(self, state):
        if not state.crashed and not state.landed:
            view = self.camera(state).compute_view(state.position)
            x, y = view.shape
            state.map[:x, :y, 3] &= ~view
            state.coverage[:x, :y] |= view

        if state.landed:
            decomp = state.decomp_init & ~state.map[..., 3]
            if np.any(decomp):
                state.decomp.append(decomp)
                state.decomp_init = state.map[..., 3].copy()

    def get_rewards(self, state):
        rewards = super().get_rewards(state)
        next_cells_remaining = self.get_remaining_cells(state)
        cells_collected = state.cells_remaining - next_cells_remaining
        state.cells_remaining = next_cells_remaining
        rewards += cells_collected * self.params.rewards.cell_reward
        if self.objective_complete(state) and state.landed:
            rewards += self.params.rewards.completion_reward
        return rewards

    @staticmethod
    def get_remaining_cells(state):
        return np.sum(state.map[..., 3])

    def objective_complete(self, state):
        return self.get_remaining_cells(state) == 0

    def get_layers(self, state, params):
        layers = super().get_layers(state, params)
        draw_what = params.terminal_state if state.terminated else params.normal_state
        env_shape = (params.env_pixels, params.env_pixels)

        if draw_what.draw_decomp:
            decomp, labels = self.draw_decomp(state, env_shape)
            layers["decomp"] = {"position": (0, 0), "canvas": decomp}
            layers["decomp_labels"] = {"position": (0, 0), "canvas": labels}
        if draw_what.draw_view:
            layers["view"] = {"position": (0, 0), "canvas": self.draw_view(state, env_shape)}

        return layers

    def draw_map(self, state, canvas_shape):
        shape = self.map_image(state).original_shape
        pix_size = canvas_shape[0] / shape[0]
        canvas = pygame.Surface(canvas_shape, pygame.SRCALPHA, 32)
        canvas.fill((255, 255, 255))
        for x in range(shape[0]):
            for y in range(shape[1]):
                cell = state.map[x, y]
                covered = state.coverage[x, y]
                init_target = state.init.target[x, y]
                pos = np.array((x, y))
                pos_image = pos * pix_size
                dim_factor = 1.0 if covered else 0.6
                patch = pygame.Rect(pos_image, (pix_size + 1, pix_size + 1))
                if cell[1] or init_target or cell[0]:
                    color = (230 * cell[1], 230 * init_target, 230 * cell[0])
                    pygame.draw.rect(canvas, np.array(color) * dim_factor, patch)
                else:
                    pygame.draw.rect(canvas, np.array((255, 255, 255)) * dim_factor, patch)

        obstacles = self.draw_obstacles(state, canvas_shape, pix_size)
        canvas.blit(obstacles, (0, 0))

        return canvas

    def draw_view(self, state, canvas_shape):
        shape = self.map_image(state).original_shape
        pix_size = canvas_shape[0] / shape[0]

        if state.landed:
            return None

        view = self.camera(state).compute_view(state.position)
        canvas = draw_shape_matrix(view, canvas_shape, pix_size, stroke=[0, 0, 255],
                                   stroke_width=int(pix_size // 4))

        return canvas

    def draw_decomp(self, state, canvas_shape):
        shape = self.map_image(state).original_shape
        pix_size = canvas_shape[0] / shape[0]

        canvas = pygame.Surface(canvas_shape, pygame.SRCALPHA, 32)
        label_canvas = pygame.Surface(canvas_shape, pygame.SRCALPHA, 32)

        colors = color_palette()
        font = pygame.font.SysFont('Arial', int(pix_size * 3))

        for k, decomp in enumerate(state.decomp):
            color = (np.array(colors[k % len(colors)]) * 255).astype(int).tolist()
            decomp_canv = draw_shape_matrix(decomp, canvas_shape, pix_size, fill=color + [102], stroke=color,
                                            stroke_width=int((pix_size + 5) // 6 * 2))
            canvas.blit(decomp_canv, (0, 0))

            idx = np.transpose(np.nonzero(decomp))
            mean = np.mean(idx, axis=0, keepdims=True)
            closest = np.argmin(np.sum(np.square(idx - mean), axis=1))
            center = idx[closest] * pix_size
            pos = center + (np.array((pix_size, pix_size)) / 2)
            draw_text(label_canvas, font, f"{k + 1}", pos, stroke=(0, 0, 0), stroke_width=2,
                      fill=color)

        return canvas, label_canvas

    def get_info(self, state=None):
        if state is None:
            state = self.state
        info = super().get_info(state)
        target_sum = np.sum(state.init.target)
        collection_ratio = (1 - self.get_remaining_cells(state) / target_sum) if target_sum > 0 else 1.0
        info.update({
            "collection_ratio": collection_ratio,
            "task_solved": self.task_solved(state) and state.landed
        })
        if self.task_solved(state) and state.landed:
            info["completion_steps"] = state.steps
            info["completion_battery"] = int(state.budget)
        return info

    def camera(self, state):
        return self._camera[state.map_index]
