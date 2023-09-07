import copy
import json
import os
import pickle
import time
from dataclasses import dataclass
from typing import Tuple, List, Union, Optional

import cv2
import numpy as np
import pygame
from tqdm import tqdm

from src.base.heuristics import GreedyHeuristic
from utils import get_value_user, NpEncoder, find_scenario, find_config_model
from src.gym.utils import draw_text, Map
import tensorflow as tf


class PyGameHuman:
    def __init__(self, key_action_mapping: List[Tuple[int, int]]):
        self.key_action_mapping = key_action_mapping

    def get_action_non_blocking(self) -> Optional[int]:
        keys = pygame.key.get_pressed()
        for key, action in self.key_action_mapping:
            if keys[key]:
                return action
        return None


class Evaluator:
    @dataclass
    class Params:
        stochastic: bool = False
        eval_maps: Union[int, Tuple[int]] = -1  # indexes of maps or -1 for all

    def __init__(self, params: Params, trainer, gym):
        self.params = params
        self.trainer = trainer
        self.gym = gym

        self.use_heuristic = False
        self.stochastic = self.params.stochastic
        self.heuristic = GreedyHeuristic(self.gym)

    def evaluate_episode(self, init=None, map_name=None):
        if map_name is None:
            eval_maps = self.params.eval_maps
            if isinstance(eval_maps, int):
                if eval_maps == -1:
                    map_index = np.random.randint(self.gym.num_maps)
                else:
                    map_index = eval_maps
            else:
                map_index = np.random.choice(eval_maps)
            map_name = self.gym.map_names[map_index]

        gym_state = self.gym.create_state()

        state, info = self.gym.reset(state=gym_state, init=init, map_name=map_name)
        while not info["timeout"]:
            action = self.get_action(state, gym_state=gym_state)
            state, reward, terminal, truncated, info = self.gym.step(state=gym_state, action=action)
            if terminal:
                break

        init = gym_state.init
        info_heur = self.evaluate_episode_heuristic(init)
        if "completion_steps" in info and "completion_steps" in info_heur:
            info["completion_step_ratio"] = info["completion_steps"] / info_heur["completion_steps"]
        info["total_step_ratio"] = info["total_steps"] / info_heur["total_steps"]

        return info

    def evaluate_episodes(self, inits, n):
        num_inits = len(inits)
        assert n <= num_inits, "Set n < len(inits)"

        gym_states = [self.gym.create_state() for _ in range(n)]

        gym_init_ids = list(range(n))
        next_init = n
        states = [self.gym.reset(state=gym_state, init=init)[0] for gym_state, init in zip(gym_states, inits)]
        state = {key: np.concatenate([s[key] for s in states], axis=0) for key in states[0].keys()}

        obs = self.trainer.observation_function(state)

        status = [None] * num_inits
        active = [True] * n
        bar = tqdm(total=num_inits)
        while any(active):

            actions, _ = self.trainer.get_action(obs, greedy=not self.stochastic)

            next_states, reward, terminated, truncated, infos = zip(
                *[self.gym.step(state=gym_state, action=np.expand_dims(actions[k], 0)) for k, gym_state in
                  enumerate(gym_states)])
            next_states = list(next_states)

            for l in range(n):
                if not active[l]:
                    continue
                terminal = terminated[l]
                info = infos[l]
                if terminal or info["timeout"]:
                    bar.update(1)
                    status[gym_init_ids[l]] = info
                    if next_init >= num_inits:
                        active[l] = False
                        continue
                    state, _ = self.gym.reset(state=gym_states[l], init=inits[next_init])
                    gym_init_ids[l] = next_init
                    next_init += 1
                    next_states[l] = state

            state = {key: np.concatenate([s[key] for s in next_states], axis=0) for key in next_states[0].keys()}
            obs = self.trainer.observation_function(state)

        return status

    def evaluate_episode_heuristic(self, init):
        gym_state = self.gym.create_state()
        state, info_heur = self.gym.reset(state=gym_state, init=init)
        while not info_heur["timeout"]:
            action = self.heuristic.get_action(state, state=gym_state)
            state, reward, terminal, truncated, info_heur = self.gym.step(state=gym_state, action=action)
            if np.all(terminal):
                break
        return info_heur

    def get_action(self, state, gym_state=None):
        if self.use_heuristic:
            return self.heuristic.get_action(state, state=gym_state)
        if self.trainer is None:
            return None
        obs = self.trainer.observation_function(state)
        return self.trainer.get_action(obs, greedy=not self.stochastic)[0]

    def evaluate_multiple_episodes(self, n):
        stats = []
        for k in tqdm(range(n)):
            stats.append(self.evaluate_episode(map_name=self.gym.map_names(k % self.gym.num_maps)))
        return stats


class InteractiveEvaluator(Evaluator):
    @dataclass
    class Params:
        stochastic: bool = False
        eval_maps: Union[int, Tuple[int]] = -1  # indexes of maps or -1 for all

    def __init__(self, params: Params, trainer, gym, human):
        super().__init__(params, trainer, gym)

        self.human = human
        self.human_mask_lvl = gym.mask_levels.index("invariant")
        self.mode = "human"  # ["run", "run_until_end", "blind", "human"]
        self.recorder = None

        self.previous_stats = None
        self.render_params = copy.deepcopy(gym.params.rendering)
        self.show_overlay = False
        self.show_help = False
        self.overlay_texts = [""]
        self.loaded_models = {}
        self.render_params.render = True
        gym.params.rendering.render = False

        gym.register_render(self.draw_overlay, (self.render_params.env_pixels, self.render_params.env_pixels),
                            offset=(0, 0))

        gym.register_render(self.draw_previous_stats,
                            (self.render_params.env_pixels // 4, self.render_params.env_pixels // 2),
                            offset=(self.render_params.env_pixels, self.render_params.env_pixels // 2))

    def evaluate_interactive(self, init=None):

        while True:
            state, info = self.gym.reset(init=init)
            self.gym.render(params=self.render_params)
            skip_show = False
            while not info["timeout"]:
                action = None
                terminate = False
                events = pygame.event.get()
                for event in events:
                    if event.type == pygame.KEYDOWN:
                        if event.type == pygame.QUIT:
                            exit(0)
                        if self.mode == "human":
                            action = self.human.get_action_non_blocking()
                            if action is not None and action != -1:
                                mask = self.gym.get_action_mask_lvl(self.gym.state, self.human_mask_lvl)
                                if mask[action]:
                                    break
                                action = None
                        keys = pygame.key.get_pressed()
                        new_init = self.reset_specific_map(keys)
                        if new_init is not None:
                            init = new_init
                            terminate = True
                            skip_show = True

                        if self.update_with_key(keys):
                            self.gym.render(params=self.render_params)
                        if keys[pygame.K_y]:
                            self.mode = "run_until_end"
                        elif keys[pygame.K_s]:
                            if self.mode == "blind":
                                self.gym.render(params=self.render_params)
                            self.mode = "human"
                        elif keys[pygame.K_u]:
                            self.mode = "blind"
                        elif keys[pygame.K_r]:
                            self.mode = "run"

                if terminate:
                    break
                if self.mode == "human":
                    # Block until a button is pressed
                    if action is None:
                        continue
                    action = np.array((action,))

                if self.mode != "human" or action < 0:
                    action = self.get_action(state)

                if action is None:
                    self.draw_message_blocking("Not Available", t=1.0)
                    self.mode = "human"
                    continue

                state, reward, terminal, truncated, info = self.gym.step(action)

                if self.mode != "blind" or terminal or info["timeout"]:
                    self.gym.render(params=self.render_params)
                if terminal:
                    break
            if self.mode == "run":
                continue
            self.mode = "human"

            if skip_show:
                continue
            terminate = False
            while not terminate:
                events = pygame.event.get()
                for event in events:
                    if event.type == pygame.QUIT:
                        exit(0)
                    if event.type == pygame.KEYDOWN:
                        keys = pygame.key.get_pressed()
                        if self.update_with_key(keys):
                            self.gym.render(params=self.render_params)
                        if keys[pygame.K_y]:
                            terminate = True
                            init = None
                        elif keys[pygame.K_r]:
                            terminate = True
                            init = self.gym.state.init
                        else:
                            new_init = self.reset_specific_map(keys)
                            if new_init is not None:
                                init = new_init
                                terminate = True

    def draw_message_blocking(self, message, t: Optional[float] = 2.0):
        self.show_overlay = True
        self.overlay_texts = [message]
        self.gym.render(params=self.render_params)
        ack = False
        start = time.time()
        while not ack and (t is None or time.time() - start < t):
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    exit(0)
                if event.type == pygame.KEYDOWN:
                    ack = True

        self.show_overlay = False
        self.gym.render(params=self.render_params)

    def get_user_input(self, query):
        self.show_overlay = True
        self.overlay_texts = [query, ""]
        self.gym.render(params=self.render_params)
        while True:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    exit(0)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_BACKSPACE:
                        if len(self.overlay_texts[1]) > 0:
                            self.overlay_texts[1] = self.overlay_texts[1][:-1]
                    elif event.key == pygame.K_KP_ENTER or event.key == pygame.K_RETURN:
                        self.show_overlay = False
                        self.gym.render(params=self.render_params)
                        return self.overlay_texts[1]
                    elif event.key == pygame.K_ESCAPE:
                        self.show_overlay = False
                        self.gym.render(params=self.render_params)
                        return None
                    else:
                        self.overlay_texts[1] += event.unicode
                    self.gym.render(params=self.render_params)

    def reset_specific_map(self, keys):
        init = None
        if keys[pygame.K_l]:
            if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                scenario = self.get_user_input("Save Scenario to 'example/scenarios/[]_init.pickle':")
                if scenario is not None:
                    with open(f"example/scenarios/{scenario}_init.pickle", "wb") as f:
                        pickle.dump(self.gym.state.init, f)
                        self.draw_message_blocking(f"Saved as example/scenarios/{scenario}_init.pickle")
            else:
                scenario = self.get_user_input("Load Scenario")
                if scenario is not None:
                    init_file = find_scenario(scenario, exit_on_not_found=False)
                    if init_file is None:
                        self.draw_message_blocking(f"File {scenario} not found")
                    else:
                        with open(init_file, 'rb') as f:
                            init = pickle.load(f)
                        print(f"Loaded from {init_file}")

        for idx, key in enumerate(range(pygame.K_1, pygame.K_9 + 1)):
            if keys[key]:
                if self.gym.num_maps <= idx:
                    self.draw_message_blocking(f"No map with index {idx}")
                else:
                    init = self.gym.generate_init(map_name=self.gym.map_names[idx])
                    break
        if keys[pygame.K_0]:
            if self.gym.num_maps <= 9:
                self.draw_message_blocking(f"No map with index {9}")
            else:
                init = self.gym.generate_init(map_name=self.gym.map_names[9])
        return init

    def update_with_key(self, keys):
        state_render = self.render_params.terminal_state if self.gym.state.terminated else self.render_params.normal_state
        if keys[pygame.K_q]:
            exit(0)
        elif keys[pygame.K_h]:
            self.show_help = True
            self.draw_message_blocking("", t=None)
            self.show_help = False
        elif keys[pygame.K_o]:
            if self.trainer is None:
                self.draw_message_blocking("Not Available", t=1.0)
                return False
            agent = self.get_user_input("Load Agent (Same architecture)")
            if agent is not None:
                agent_config = find_config_model(agent, exit_on_not_found=False)
                if agent_config is None:
                    self.draw_message_blocking(f"Agent {agent} not found")
                else:
                    model_dir = agent_config.rsplit('/', maxsplit=1)[0] + "/models"

                    if model_dir in self.loaded_models:
                        actor, critic = self.loaded_models[model_dir]
                    else:
                        actor = tf.keras.models.load_model(f"{model_dir}/actor.keras")
                        critic = tf.keras.models.load_model(f"{model_dir}/critic.keras")
                        self.loaded_models[model_dir] = (actor, critic)
                    self.trainer.agent.actor.model.set_weights(actor.get_weights())
                    self.trainer.agent.critic.model.set_weights(critic.get_weights())
        elif keys[pygame.K_g]:
            self.use_heuristic = not self.use_heuristic
            self.draw_message_blocking("Switched to Heuristic" if self.use_heuristic else "Switched to Agent", t=1.0)
        elif keys[pygame.K_t]:
            self.stochastic = not self.stochastic
            self.draw_message_blocking("Stochastic actions" if self.stochastic else "Greedy actions", t=1.0)
        elif keys[pygame.K_e]:
            self.human_mask_lvl = (self.human_mask_lvl + 1) % len(self.gym.mask_levels)
            self.draw_message_blocking(f"Action Mask: {self.gym.mask_levels[self.human_mask_lvl]}", t=1.0)
        elif keys[pygame.K_b]:
            state_render.draw_trajectory = not state_render.draw_trajectory
            return True
        elif keys[pygame.K_d]:
            state_render.draw_decomp = not state_render.draw_decomp
            return True
        elif keys[pygame.K_v]:
            state_render.draw_view = not state_render.draw_view
            return True
        elif keys[pygame.K_j]:
            self.start_editor()
            return True
        elif keys[pygame.K_k]:
            new_info = self.gym.get_info(self.gym.state)
            if new_info != self.previous_stats:
                self.previous_stats = new_info
            else:
                self.previous_stats = None
            return True
        elif keys[pygame.K_w]:
            small = not pygame.key.get_mods() & pygame.KMOD_SHIFT
            directory_name = "./screenshots"
            filename = self.get_user_input(
                f"Taking screenshot {'of map only ' if small else ''}: Filename {directory_name}/[].png?")
            if filename is None:
                return False
            frame = pygame.surfarray.pixels3d(self.gym.window)
            frame = np.transpose(frame, (1, 0, 2))
            frame = frame[..., ::-1]
            if small:
                frame = frame[:, :frame.shape[0]]
            if not os.path.exists(directory_name):
                os.makedirs(directory_name)
            cv2.imwrite(f"{directory_name}/{filename}.png", frame)
            del frame
            self.draw_message_blocking(f"Saved as {directory_name}/{filename}.png", t=2)
        return False

    def draw_overlay(self, canvas_shape, obs):

        if not self.show_overlay:
            return None
        if self.show_help:
            return self.draw_help(canvas_shape)
        canvas = pygame.Surface(canvas_shape, pygame.SRCALPHA, 32)
        text_canvas = pygame.Surface(canvas_shape, pygame.SRCALPHA, 32)

        font_size = 20
        font_dist = 30
        font = pygame.font.SysFont('Arial', font_size, bold=True)
        draw_pos = np.array(canvas_shape) / 2
        height = len(self.overlay_texts) * font_dist
        offset = (0, height / 2)
        width = 100

        for k, text in enumerate(self.overlay_texts):
            s = draw_text(text_canvas, font, text, draw_pos - offset + (0, font_dist * k))
            width = max(width, s[0] + 20)

        canvas.fill((0, 0, 0, 100))
        height += 20
        rect = pygame.Rect(np.array(canvas_shape) / 2 - [width // 2, height // 2], [width, height])
        pygame.draw.rect(canvas, (200, 200, 200, 150), rect)
        canvas.blit(text_canvas, (0, 0))

        return canvas

    def draw_previous_stats(self, canvas_shape, obs):

        if self.previous_stats is None or not self.render_params.draw_stats:
            return None

        stats_canvas = self.gym.draw_specific_stats(canvas_shape, self.previous_stats)
        canvas = pygame.Surface(canvas_shape, pygame.SRCALPHA, 32)
        draw_text(canvas, pygame.font.SysFont('Arial', 25, bold=True), "Previous Stats", (0, 0), fill=(255, 255, 255),
                  align="top left")
        canvas.blit(stats_canvas, (0, 30))

        return canvas

    def draw_help(self, canvas_shape):
        canvas = pygame.Surface(canvas_shape, pygame.SRCALPHA, 32)
        canvas.fill((0, 0, 0, 100))

        text_canvas_shape = np.array(canvas_shape) / 1.5
        text_canvas = pygame.Surface(text_canvas_shape, pygame.SRCALPHA, 32)
        text_canvas.fill((200, 200, 200, 230))

        cent_x = text_canvas_shape[0] / 100
        cent_y = text_canvas_shape[1] / 100

        header_font = pygame.font.SysFont('Arial', int(cent_y * 5), bold=True)
        items_font = pygame.font.SysFont('Arial', int(cent_y * 3), bold=False)

        draw_text(text_canvas, header_font, "Help", (50 * cent_x, 5 * cent_y))
        action_grid = self.gym.draw_action_grid(["RIGHT", "DOWN", "LEFT", "UP", "SPACE", "m", "n"],
                                                tile_size=10 * cent_x)
        text_canvas.blit(action_grid, (40 * cent_x, 10 * cent_y))
        draw_text(text_canvas, header_font, "Actions:", (35 * cent_x, 10 * cent_y + 15 * cent_x), align="right")

        bindings = {
            "y": "run until end / new run",
            "u": "run blind until end",
            "r": "run",
            "s": "human control / step",
            "e": "toggle mask human",
            "v": "toggle view",
            "b": "toggle trajectory",
            "d": "toggle decomposition",
            "g": "toggle heuristic",
            "t": "toggle stochastic",
            "k": "keep stats",
            "o": "load agent",
            "l": "load scenario",
            "w": "screenshot (map only)",
            "shift + w": "screenshot",
            "shift + l": "store scenario",
            "1-(1)0": "init map index",
            "j": "start editor (experimental)",
            "q": "quit"
        }

        start_y = 45 * cent_y
        num = len(bindings)
        for k, (key, effect) in enumerate(bindings.items()):
            col = 0 if k < num / 2 else 1
            shift = 50 * cent_x * col
            k_y = k - col * np.ceil(num / 2)
            draw_text(text_canvas, items_font, f"{key}:", (shift + 5 * cent_x, k_y * 4 * cent_y + start_y),
                      align="left")
            draw_text(text_canvas, items_font, effect, (shift + 45 * cent_x, k_y * 4 * cent_y + start_y), align="right")

        draw_text(text_canvas, header_font, "Press any key to continue", (50 * cent_x, 90 * cent_y))

        canvas.blit(text_canvas, np.array(canvas_shape) / 6)

        return canvas

    def record_episode(self, init, name, fps=15):
        self.gym.params.rendering.render = True

        for dir_name in ["screenshots", "videos", "stats"]:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

        old_use_pygame = self.gym.use_pygame_display
        old_window = self.gym.window
        window_shape = self.gym.get_window_shape(self.render_params)

        canvas = pygame.Surface(window_shape)
        self.gym.window = canvas

        state, info = self.gym.reset(init=init)

        recorder = cv2.VideoWriter(f"./videos/{name}.mp4", cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps, window_shape)

        while not info["timeout"]:
            frame = self.get_frame(map_only=False, canvas=canvas)
            recorder.write(frame)
            del frame

            action = self.get_action(state)
            state, reward, terminal, truncated, info = self.gym.step(action)
            if terminal:
                break
        frame = self.get_frame(map_only=False, canvas=canvas)
        for _ in range(fps):
            recorder.write(frame)
        del frame
        recorder.release()

        frame = self.get_frame(map_only=True, canvas=canvas)
        cv2.imwrite(f"./screenshots/{name}.png", frame)
        del frame

        with open(f"./stats/{name}.json", 'w') as f:
            json.dump(info, f, cls=NpEncoder, indent=4)

        self.gym.window = old_window
        self.gym.use_pygame_display = old_use_pygame

    def draw_maps(self):
        self.gym.params.rendering.render = False
        for m in self.gym.map_names:
            init = self.gym.generate_init(map_name=m)
            init.target = np.zeros_like(init.target)
            self.gym.reset(init=init)
            self.gym.state.map[..., 4] = True

            canvas = self.gym.draw_map(self.gym.state, (800, 800))
            frame = self.get_frame(True, canvas)
            cv2.imwrite(f"./{m}_raw.png", frame)
            del frame

    def get_frame(self, map_only, canvas=None):
        if canvas is None:
            canvas = self.gym.window
        frame = pygame.surfarray.pixels3d(canvas)
        frame = np.transpose(frame, (1, 0, 2))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if map_only:
            frame = frame[:, :frame.shape[0]]
        return frame

    def start_editor(self):
        running = True
        old_map_size = copy.deepcopy(self.gym.map_image().original_shape)
        map_size = np.array(old_map_size)
        pix = np.array((self.render_params.env_pixels, self.render_params.env_pixels))
        mouse_pressed = False
        shift_held = False
        channel = 3

        old_env = self.gym.state.map[..., :3].copy()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        if np.not_equal(old_env, self.gym.state.map[..., :3]).any():
                            map_name = self.get_user_input("Map changed, store as new map? res/[].png")
                            if map_name is not None:
                                filename = f"res/{map_name}.png"
                                map_data = self.gym.state.map[:map_size[0], :map_size[1], [0, 2, 1]].transpose(1, 0, 2)
                                cv2.imwrite(filename, (map_data * 255).astype(int))
                                if self.gym.add_map(filename) != -1:
                                    running = False
                                    self.gym.map_image().original_shape = old_map_size
                                    self.gym.state.init.target = self.gym.state.map[:map_size[0], :map_size[1], 3]
                                    self.gym.state.init.map_name = map_name
                                    self.gym.reset(init=self.gym.state.init)
                                else:
                                    print("Map not solvable, fix")
                            else:
                                cont = self.get_user_input("continue? yes: enter, no: escape")
                                running = cont is not None
                        else:
                            self.gym.state.init.target = self.gym.state.map[:map_size[0], :map_size[1], 3]
                            self.gym.reset(init=self.gym.state.init)
                            running = False
                    elif event.key == pygame.K_n:
                        size = self.get_user_input("New map size")
                        if size is None:
                            break
                        size = int(size)
                        self.gym.map_image().original_shape = (size, size)
                        old_map_size = (size, size)
                        self.gym.state.map[:size, :size, :] = False
                        map_size = np.array(self.gym.map_image().original_shape)
                    elif pygame.K_0 <= event.key <= pygame.K_9:
                        number = event.key - pygame.K_0  # Calculate the pressed number
                        if 0 <= number < self.gym.state.map.shape[2]:
                            channel = number
                    elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                        shift_held = True
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                        shift_held = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pressed = True
                if event.type == pygame.MOUSEBUTTONUP:
                    mouse_pressed = False

            if mouse_pressed:
                mouse = np.array(pygame.mouse.get_pos())
                cell = np.floor(mouse / pix * map_size).astype(int)
                if 0 <= cell[0] < map_size[0] and 0 <= cell[1] < map_size[1]:
                    if shift_held:
                        self.gym.state.map[cell[0], cell[1], channel] = False
                    else:
                        cell_val = self.gym.state.map[cell[0], cell[1]]
                        if not (channel == 3 and cell_val[2] or channel == 2 and cell_val[
                            3] or channel == 0 and (cell_val[1] or cell_val[2]) or channel == 1 and
                                cell_val[0] or channel == 2 and cell_val[0]):
                            self.gym.state.map[cell[0], cell[1], channel] = True

            self.gym.last_map_id = -1
            self.gym.render(params=self.render_params)
