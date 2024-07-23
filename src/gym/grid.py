import copy
from dataclasses import dataclass
from typing import Optional, Union, List, Callable, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import pygame

from src.gym.utils import Map, get_arrow_polygon, draw_text, draw_shape_matrix

from src.trainer.observation import ObservationFunctionFactory


@dataclass
class GridStateRenderParams:
    draw_trajectory: bool = False


@dataclass
class GridRenderParams:
    render: bool = False
    draw_stats: bool = False
    render_fps: int = 0
    env_pixels: int = 768
    addons: bool = True

    normal_state: GridStateRenderParams = GridStateRenderParams()
    terminal_state: GridStateRenderParams = GridStateRenderParams()


@dataclass
class GridRewardParams:
    boundary_penalty: float = 1.0
    empty_battery_penalty: float = 150.0
    movement_penalty: float = 0.2


class GridGym(gym.Env):
    @dataclass
    class Params:
        map_path: Union[str, List[str]] = "res/manhattan32_mod.png"
        min_size: int = 50  # Maps are padded to this size if they are smaller
        add_rotated_maps: bool = False
        budget_range: (int, int) = (50, 75)
        start_landed: bool = True
        safety_controller: bool = True  # If True, actions that cause an immediate crash are rejected
        timeout_steps: Union[int, List[int]] = 1000

        # Charging
        recharge: bool = True
        charge_amount: float = 2.0

        rewards: GridRewardParams = GridRewardParams()

        rendering: GridRenderParams = GridRenderParams()

        action_masking: str = "invariant"  # ["none", "invalid", "immediate", "invariant"]

        # Position History
        position_history_alpha: float = 0.99

        # Observation function
        obs_function: ObservationFunctionFactory.default_param_type() = ObservationFunctionFactory.default_params()

    @dataclass
    class Init:
        position: np.ndarray
        budget: int
        map_name: str

    @dataclass
    class State:
        init = None

        map_index = 0
        map = None
        position_history = None

        action_mask = None

        position = np.zeros(2)
        budget = 0
        infeasible = False
        crashed = False
        landed = False
        terminated = False
        truncated = False

        boundary_counter = 0
        charging_steps = 0
        episodic_reward = 0
        steps = 0

        trajectory = []

    def __init__(self, params: Params):
        pygame.init()
        pygame.font.init()
        self.params = params

        self.shape = np.zeros((2,))
        self._map_image = []
        self.map_array = None
        self.map_path = params.map_path
        if not isinstance(self.map_path, list):
            self.map_path = [self.map_path]
        self.load_maps(self.map_path)

        self._timeout_steps = self.params.timeout_steps
        if not isinstance(self._timeout_steps, list):
            self._timeout_steps = [self._timeout_steps] * self.num_maps
        assert len(self._timeout_steps) == self.num_maps, "Num Maps != Num timeouts & Num timeouts != 1"

        self.action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
            4: np.array([0, 0]),
            5: np.array([0, 0]),
            6: np.array([0, 0])
        }

        self.action_to_name = {
            0: "right",
            1: "down",
            2: "left",
            3: "up",
            4: "land",
            5: "take off",
            6: "charge"
        }

        self.state = self.create_state()

        num_actions = len(self.action_to_direction)
        self.action_space = spaces.Discrete(num_actions)

        self.observation_function = ObservationFunctionFactory.create(params.obs_function,
                                                                      max_budget=params.budget_range[-1])

        self.mask_levels = ["none", "valid", "immediate", "invariant"]
        self.masking_level = self.mask_levels.index(self.params.action_masking)

        self.map_layers = ["landing", "nfz", "obstacle"]

        self.render_registry = []
        self.render_registry_offset = 0
        self.layer_order = ["environment", "trajectory", "agent"]
        self.last_obst = np.zeros(self.shape)
        self.obstacle_canvas = None
        self.use_pygame_display = False
        self.window = None
        self.clock = None

    def close(self):
        pass

    def step_multi(self, states, actions):

        rewards = [self._step(state, action) for state, action in zip(states, actions)]
        observes = self.observation_function.observe_multi(states)
        reward = np.array(rewards)
        terminated = np.array([state.terminated for state in states])
        truncated = np.array([state.truncated for state in states])
        infos = [self.get_info(state) for state in states]

        return observes, reward, terminated, truncated, infos

    def step(self, action, state=None):
        if state is None:
            state = self.state

        reward = self._step(state, action)

        self.render(state=state)

        return self.get_obs(state), reward, state.terminated, state.truncated, self.get_info(state)

    def _step(self, state, action):
        if isinstance(action, np.ndarray):
            action = action[0]
        self.motion_step(state, action)
        reward = self.get_rewards(state)
        state.episodic_reward += reward
        state.truncated = state.steps >= self.timeout_steps(state)
        return reward

    def get_rewards(self, state):
        rewards = -self.params.rewards.movement_penalty
        if state.infeasible:
            rewards -= self.params.rewards.boundary_penalty
        if state.crashed:
            rewards -= self.params.rewards.empty_battery_penalty
        return rewards

    def motion_step(self, state, action):

        # Evaluate individual motions
        if state.crashed:
            return
        if action == 4:
            if not self.can_land(state):
                state.infeasible = True
            else:
                state.landed = True
        elif action == 5:
            if state.landed:
                state.landed = False
                state.infeasible = False
            else:
                state.infeasible = True
        elif action == 6:
            if self.params.recharge and state.landed:
                # Already landed and charging
                state.budget += self.params.charge_amount + 1  # +1 for movement subtraction
                state.budget = min(state.budget, self.params.budget_range[1] + 1)  # Constrain to max battery
                state.charging_steps += 1
            else:
                state.infeasible = True
        else:
            if state.landed:
                state.infeasible = True
            else:
                motion = self.action_to_direction[action]
                idx = state.position + motion
                # Check if action is safe
                state.infeasible = not self.in_bounds(idx) or state.map[idx[0], idx[1], 1] or (
                        state.map[idx[0], idx[1], 0] and state.map[idx[0], idx[1], 2])
                if not state.infeasible:
                    # Apply motion
                    state.position += motion

        state.position_history *= self.params.position_history_alpha
        state.position_history[state.position[0], state.position[1]] = 1

        state.boundary_counter += int(state.infeasible)
        # Always consume battery
        state.budget -= 1
        state.budget = max(state.budget, 0)
        state.crashed = state.budget <= 0 and not state.landed
        if not self.params.safety_controller:
            state.crashed = state.crashed or state.infeasible
        state.infeasible = state.infeasible or state.crashed
        state.steps += 1
        state.trajectory.append([copy.deepcopy(state.position), copy.deepcopy(state.landed)])

        state.terminated = state.crashed or state.landed and self.objective_complete(state)

        state.action_mask = self.get_action_masks(state)

    @staticmethod
    def can_land(state):
        can_land = state.map[state.position[0], state.position[1], 0] and not state.landed
        return can_land

    def reset(self, state=None, seed=None, options=None, init=None, map_index=None):
        if state is None:
            state = self.state

        if init is None:
            init = self.generate_init(map_index)
        state.init = init
        self._initialize_map(state, init.map_index)
        self.reset_state(state, init)

        self.render(state=state)

        return self.get_obs(state), self.get_info(state)

    def reset_state(self, state, init: Init):
        state.position = init.position.copy()
        state.budget = init.budget
        state.infeasible = False
        state.crashed = False
        state.landed = self.params.start_landed
        state.terminated = False
        state.boundary_counter = 0
        state.charging_steps = 0
        state.episodic_reward = 0
        state.steps = 0
        state.action_mask = self.get_action_masks(state)
        state.trajectory = [[copy.deepcopy(state.position), copy.deepcopy(state.landed)]]

        state.position_history = np.zeros_like(state.map[..., 0], dtype=float)
        state.position_history[state.position[0], state.position[1]] = 1

    def generate_init(self, map_name=None) -> Init:
        if map_name is not None:
            map_index = self.get_map_index(map_name)
        else:
            map_index = np.random.randint(len(self._map_image))
            map_name = self.map_names[map_index]
        position_mask = (self._map_image[map_index].slz & ~self._map_image[map_index].obst).astype(float)
        candidates = np.argsort(np.reshape(np.random.uniform(0, 1, size=position_mask.shape) * position_mask, (-1,)))
        position = np.transpose(np.array(np.unravel_index(candidates[-1], position_mask.shape)))
        budget = np.round(np.random.uniform(low=self.params.budget_range[0], high=self.params.budget_range[1],
                                            size=1))[0]
        return GridGym.Init(position=position, budget=budget, map_name=map_name)

    def get_map_index(self, map_name):
        map_index = self.map_names.index(map_name)
        return map_index

    def objective_complete(self, state):
        return True

    def task_solved(self, state):
        return state.landed and self.objective_complete(state)

    def get_obs(self, state):
        return self.observation_function.observe(state)

    def get_action_masks(self, state):

        level = self.masking_level
        return self.get_action_mask_lvl(state, level)

    def get_action_mask_lvl(self, state, level):
        mask = np.ones(len(self.action_to_direction), dtype=bool)
        # Invalid masking
        if level > 0:
            if state.landed:
                mask[:4] = False
                mask[4] = False
                mask[6] = state.budget < self.params.budget_range[1]
            else:
                mask[4] = state.map[state.position[0], state.position[1], 0]
                mask[5] = False
                mask[6] = False
            # Immediate Masking
            if level > 1:
                for a in range(4):
                    target = state.position + self.action_to_direction[a]
                    if not self.in_bounds(target):
                        mask[a] = False
                    elif state.map[target[0], target[1], 1] or (state.map[target[0], target[1], 0] and state.map[target[0], target[1], 2]):
                        mask[a] = False
                    elif level > 2:
                        lm = self.landing_map(state)
                        if (target >= lm.shape[:2]).any() or state.budget <= lm[target[0], target[1]]:
                            mask[a] = False
                # Invariant Masking
                if level > 2:
                    if state.budget < 2:
                        mask[5] = False  # Cannot take off with too little battery

        return mask

    def get_info(self, state=None):
        if state is None:
            state = self.state
        return {
            "landed": state.landed,
            "crashed": state.crashed,
            "terminal": state.terminated,
            "boundary_counter": state.boundary_counter,
            "episodic_reward": state.episodic_reward,
            "task_solved": self.task_solved(state),
            "total_steps": state.steps,
            "timeout": state.steps >= self.timeout_steps(state),
            "charging_steps": state.charging_steps,
            "map_index": state.map_index,
            "map_name": self.map_name(state)
        }

    def render(self, state=None, params: Optional[GridRenderParams] = None):
        if state is None:
            state = self.state
        if params is None:
            params = self.params.rendering
        if params.render:
            return self.render_frame(state, params)

    def register_render(self,
                        render_func: Callable[[Tuple[int, int], dict], Optional[pygame.Surface]],
                        shape: [int, int],
                        offset: Optional[Tuple[int, int]] = None,
                        add_left=False):
        item = {
            "render_func": render_func,
            "shape": shape,
            "offset": offset
        }
        if add_left:
            self.render_registry.insert(0, item)
        else:
            self.render_registry.append(item)

    def get_window_shape(self, params: GridRenderParams):
        window_shape = [params.env_pixels, params.env_pixels]
        if params.draw_stats:
            window_shape[0] = max(window_shape[0], window_shape[0] + params.env_pixels // 4)
        self.render_registry_offset = window_shape[0]
        if params.addons:
            for render in self.render_registry:
                shape = render["shape"]
                offset = render["offset"]

                if offset is None:
                    window_shape[0] += shape[0]
                    window_shape[1] = max(window_shape[1], shape[1])
                else:
                    window_shape[0] = max(window_shape[0], offset[0] + shape[0])
                    window_shape[1] = max(window_shape[1], offset[1] + shape[1])

        return window_shape

    def render_frame(self, state, params):
        if self.window is None:
            pygame.display.init()
            self.use_pygame_display = True
            window_shape = self.get_window_shape(params)
            self.window = pygame.display.set_mode(window_shape)
            pygame.display.set_caption("uavSim")

        if self.clock is None:
            self.clock = pygame.time.Clock()

        layers = self.get_layers(state, params)
        self.draw_layers(self.window, layers)

        if self.use_pygame_display:
            pygame.event.pump()
            pygame.display.update()

            if params.render_fps > 0:
                self.clock.tick(params.render_fps)

    def get_layers(self, state, params):
        layers = {}
        draw_what = params.terminal_state if state.terminated else params.normal_state

        env_shape = (params.env_pixels, params.env_pixels)

        layers["environment"] = {"position": (0, 0), "canvas": self.draw_map(state, env_shape)}
        layers["agent"] = {"position": (0, 0), "canvas": self.draw_agent(state, env_shape)}

        if draw_what.draw_trajectory:
            layers["trajectory"] = {"position": (0, 0), "canvas": self.draw_trajectory(state, env_shape)}

        if params.draw_stats:
            layers["stats"] = {"position": (params.env_pixels, 0),
                               "canvas": self.draw_stats((params.env_pixels // 4, params.env_pixels), state)}

        if params.addons:
            x = self.render_registry_offset
            obs = self.get_obs(state)
            for k, render in enumerate(self.render_registry):
                shape = render["shape"]
                canvas = render["render_func"](shape, obs)
                if canvas is None:
                    continue
                if render["offset"] is not None:
                    position = render["offset"]
                else:
                    position = (x, 0)
                    x += shape[0]

                layers[f"__addon__{k}"] = {"position": position, "canvas": canvas}

        return layers

    def draw_layers(self, window, layers):
        for layer_id in self.layer_order:
            if layer_id in layers:
                layer = layers[layer_id]
                canvas = layer["canvas"]
                if canvas is None:
                    continue
                window.blit(canvas, layer["position"])
        # Draw addons
        if "stats" in layers:
            layer = layers["stats"]
            window.blit(layer["canvas"], layer["position"])

        for key, layer in layers.items():
            if "__addon__" not in key:
                continue
            window.blit(layer["canvas"], layer["position"])

    def draw_map(self, state, canvas_shape):
        shape = self.map_image(state).original_shape
        pix_size = canvas_shape[0] / shape[0]
        canvas = pygame.Surface(canvas_shape, pygame.SRCALPHA, 32)
        canvas.fill((255, 255, 255, 255))

        for x in range(shape[0]):
            for y in range(shape[1]):
                cell = state.map[x, y]
                pos = np.array((x, y))
                pos_image = pos * pix_size
                if cell[0]:
                    # Start landing zone
                    pygame.draw.rect(canvas, (0, 0, 255), pygame.Rect(pos_image, (pix_size + 1, pix_size + 1)))
                if cell[1]:
                    # NFZ
                    pygame.draw.rect(canvas, (255, 0, 0), pygame.Rect(pos_image, (pix_size + 1, pix_size + 1)))

        obstacles = self.draw_obstacles(state, canvas_shape, pix_size)
        canvas.blit(obstacles, (0, 0))

        return canvas

    def draw_obstacles(self, state, canvas_shape, pix_size):
        x, y = self.map_image(state).original_shape
        obst = state.map[:x, :y, 2]
        if not np.array_equal(obst.shape, self.last_obst.shape) or not np.array_equal(obst, self.last_obst):
            self.obstacle_canvas = pygame.Surface(canvas_shape, pygame.SRCALPHA, 32)
            self.obstacle_canvas.fill((255, 255, 255, 0))

            nfz = state.map[:x, :y, 1]
            low_obst = np.logical_and(obst, np.logical_not(nfz))
            high_obst = np.logical_and(obst, nfz)
            stroke_width = int((pix_size + 1) // 8 * 2)
            self.obstacle_canvas.blit(draw_shape_matrix(obst, canvas_shape, pix_size, fill=(0, 0, 0, 50),
                                                        stroke=(0, 0, 0),
                                                        stroke_width=stroke_width), (0, 0))
            self.obstacle_canvas.blit(draw_shape_matrix(high_obst, canvas_shape, pix_size, fill=None,
                                                        stroke=(0, 0, 0),
                                                        stroke_width=(stroke_width + 3) // 4 * 2), (0, 0))
            self.obstacle_canvas.blit(draw_shape_matrix(low_obst, canvas_shape, pix_size, fill=None,
                                                        stroke=(0, 0, 0),
                                                        stroke_width=stroke_width // 4 * 2), (0, 0))
            self.last_obst = obst
        return self.obstacle_canvas

    def draw_agent(self, state, canvas_shape):
        canvas = pygame.Surface(canvas_shape, pygame.SRCALPHA, 32)
        pix_size = self.pix_size(state)
        color = (0, 180, 0)
        if state.infeasible:
            color = (255, 180, 0)
        if state.crashed:
            color = (180, 0, 0)
        if state.landed:
            color = (0, 90, 0)
        position = state.position

        map_position = position * pix_size
        pygame.draw.circle(canvas, color, map_position + np.array((pix_size, pix_size)) / 2, radius=pix_size / 2 - 2,
                           width=0)
        pygame.draw.circle(canvas, (200, 200, 0), map_position + np.array((pix_size, pix_size)) / 2,
                           radius=pix_size / 2,
                           width=2)

        budget_text = f"{int(state.budget)}"
        font = pygame.font.SysFont('Arial', int(pix_size), bold=True)
        draw_pos = map_position + (np.array((pix_size, pix_size)) / 2)
        draw_text(canvas, font, budget_text, draw_pos, fill=(255, 255, 255), stroke=(0, 0, 0))

        return canvas

    def draw_trajectory(self, state, canvas_shape):
        canvas = pygame.Surface(canvas_shape, pygame.SRCALPHA, 32)
        pix_size = self.pix_size(state)
        positions, landed = zip(*state.trajectory)
        traj = np.array(positions)  # [t, pos]
        landed = np.array(landed)  # [t]

        takeoff = traj[0]
        pygame.draw.circle(canvas, (255, 255, 255), takeoff * pix_size + np.array((pix_size, pix_size)) / 2,
                           radius=pix_size / 2 - 5,
                           width=0)
        pygame.draw.circle(canvas, (0, 0, 0), takeoff * pix_size + np.array((pix_size, pix_size)) / 2,
                           radius=pix_size / 2,
                           width=5)
        takeoff_done = False
        for t, l in enumerate(landed):
            if not takeoff_done and not l:
                takeoff_done = True
            if takeoff_done and l:
                landing = traj[t]
                pygame.draw.circle(canvas, (255, 255, 255), landing * pix_size + np.array((pix_size, pix_size)) / 2,
                                   radius=pix_size / 2 - 5,
                                   width=0)
                pygame.draw.circle(canvas, (0, 150, 0), landing * pix_size + np.array((pix_size, pix_size)) / 2,
                                   radius=pix_size / 2,
                                   width=5)
        last_pos = traj[0]
        for pos in traj[1:]:
            if pos[0] != last_pos[0] or pos[1] != last_pos[1]:
                orig = (last_pos + 0.5) * pix_size
                dest = (pos + 0.5) * pix_size

                points = get_arrow_polygon(orig, dest)
                pygame.draw.polygon(canvas, (0, 0, 0), points)

            last_pos = pos

        return canvas

    def draw_stats(self, canvas_shape, state):

        info = self.get_info(state)
        return self.draw_specific_stats(canvas_shape, info)

    @staticmethod
    def draw_specific_stats(canvas_shape, info):
        info_texts = [(f"{key}:", f"{value:.3f}") if isinstance(value, float) else (f"{key}:", f"{value}") for
                      key, value in
                      info.items()]
        font = pygame.font.SysFont('Arial', 16)
        infos = [(font.render(name, True, (0, 0, 0)), font.render(value, True, (0, 0, 0))) for name, value in
                 info_texts]
        max_height = max([text.get_size()[1] for text, _ in infos])
        height = max_height + 5
        width = canvas_shape[0]
        stats_canvas_base = pygame.Surface(canvas_shape, pygame.SRCALPHA, 32)
        stats_canvas_base.fill((0, 0, 0))
        stats_canvas = pygame.Surface((width, height * len(info_texts)), pygame.SRCALPHA)
        stats_canvas.fill((255, 255, 255))
        for k, (name, value) in enumerate(infos):
            stats_canvas.blit(name, (1, k * height))
            stats_canvas.blit(value, (width - value.get_size()[0] - 1, k * height))
        stats_canvas_base.blit(stats_canvas, dest=(0, 0))
        return stats_canvas_base

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None

    def load_maps(self, map_paths):
        self._map_image = []
        max_size = [0, 0]
        for path in map_paths:
            m = Map.load_map(path)
            max_size = max(max_size[0], m.get_size()[0]), max(max_size[1], m.get_size()[1])
            self._map_image.append(m)
        if max_size[0] < self.params.min_size:
            max_size = [self.params.min_size] * 2
        self.pad_maps_to_size(max_size)

        if self.params.add_rotated_maps:
            old_maps = copy.deepcopy(self._map_image)
            self._map_image = []
            for m in old_maps:
                # compute all model elements or check if they are already computed
                m.all_distances()  # compute all distances
                m.landing_distances()  # compute landing distances
                m.shadowing()  # compute shadowing

                m = copy.deepcopy(m)
                m.pad_model(max_size)
                self._map_image.append(copy.deepcopy(m))
                self._map_image.append(copy.deepcopy(m.rotate_90()))
                self._map_image.append(copy.deepcopy(m.rotate_90()))
                self._map_image.append(copy.deepcopy(m.rotate_90()))
        self.shape = np.array(max_size)

        self.map_array = np.stack([m.map_data for m in self._map_image],
                                  axis=0)

    def pad_maps_to_size(self, size):
        for m in self._map_image:
            m.pad_to_size(size)

    def add_map(self, filename):
        m = Map.load_map(filename)
        m.pad_to_size(self.shape)
        if m.name in self.map_names:
            index = self.map_names.index(m.name)
            self._map_image[index] = m
        else:
            self._map_image.append(m)
            self._timeout_steps.append(1500)
            index = len(self._map_image) - 1
        return index

    def _initialize_map(self, state, map_index=None):
        if map_index is None:
            map_index = np.random.randint(len(self._map_image))
        state.map_index = map_index
        state.map = self.map_image(state).map_data[..., [2, 0, 1]]

    def initialize_map(self, state, map_index=None):
        self._initialize_map(state, map_index)

    def draw_action_grid(self, strings, tile_size):
        action_canvas = pygame.Surface((4 * tile_size, 3 * tile_size), pygame.SRCALPHA, 32)
        action_canvas.fill((0, 0, 0))
        special_actions = 0
        for a, action in self.action_to_direction.items():
            if sum(action) == 0:
                offset = np.array([3, special_actions]) * tile_size
                special_actions += 1
            else:
                offset = action * tile_size + tile_size

            tile_canvas = self.draw_tile(tile_size, self.action_to_name[a], strings[a])
            action_canvas.blit(tile_canvas, offset)

        return action_canvas

    def create_state(self):
        return GridGym.State()

    @staticmethod
    def draw_tile(size, name, value_string):
        s = (size, size)
        value_font = pygame.font.SysFont('Arial', 16)
        name_font = pygame.font.SysFont('Arial', 12)
        canvas = pygame.Surface(s, pygame.SRCALPHA, 32)
        canvas.fill((255, 255, 255))
        pygame.draw.rect(canvas, (0, 0, 0), pygame.Rect((0, 0), s), width=1)
        action_name = name_font.render(name, True, (50, 50, 50))
        canvas.blit(action_name, [1, 0])
        f_val = value_font.render(value_string, True, (0, 0, 0))
        canvas.blit(f_val, (np.array(s) / 2 - np.array(f_val.get_size()) / 2).astype(int))
        return canvas

    def map_layer_id(self, name):
        return self.map_layers.index(name)

    def map_layer(self, state, name):
        return state.map[..., self.map_layer_id(name)]

    def map_image(self, state=None):
        if state is None:
            state = self.state
        return self._map_image[state.map_index]

    def map_name(self, state=None):
        return self.map_image(state).name

    @property
    def map_names(self):
        return [map_image.name for map_image in self._map_image]

    def landing_map(self, state=None):
        return self.map_image(state).landing_distances()

    def timeout_steps(self, state=None):
        if state is None:
            state = self.state
        return self._timeout_steps[state.map_index]

    @property
    def max_budget(self):
        return self.params.budget_range[1]

    @property
    def num_maps(self):
        return len(self._map_image)

    @property
    def centered_shape(self):
        return self.shape * 2 - 1

    @property
    def observation_space(self):
        state = self.create_state()
        self.reset(state=state)
        return self.observation_function.get_observation_space(state)

    def pix_size(self, state=None):
        return self.params.rendering.env_pixels / self.map_image(state).original_shape[0]

    def in_bounds(self, target):
        if (target < 0).any():
            return False
        if (target >= self.shape).any():
            return False
        return True
