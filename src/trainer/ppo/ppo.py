import copy
from dataclasses import dataclass

import cv2
import numpy as np
import pygame
from tqdm import tqdm

import tensorflow as tf

from src.gym.utils import draw_text
from src.trainer.trainer import BaseTrainer, DiscountCurve
from src.trainer.utils import DecayParams
from utils import dict_mean

from tensorflow.keras.optimizers.schedules import ExponentialDecay


class PPOTrainer(BaseTrainer):
    @dataclass
    class Params(BaseTrainer.Params):

        rollout_length: int = 10_000
        rollout_gyms: int = 8

        lam: float = 0.7  # 0.5 - 0.9
        epsilon: float = 0.2
        beta: float = 0.001  # 0.001 - 0.1
        normalize_advantage: bool = False

        batch_size: int = 128
        rollout_epochs: int = 4

        actor_lr: DecayParams = DecayParams()
        critic_lr: DecayParams = DecayParams()

        debug_observations: bool = False

    def __init__(self, params: Params, gym, agent, observation_function, action_space, logger):
        super().__init__(gym, observation_function, action_space)
        self.params = params
        self.agent = agent
        self.logger = logger

        self.actor_optimizer = tf.keras.optimizers.Adam(ExponentialDecay(self.params.actor_lr.base,
                                                                         decay_steps=self.params.actor_lr.decay_steps,
                                                                         decay_rate=self.params.actor_lr.decay_rate))
        self.critic_optimizer = tf.keras.optimizers.Adam(ExponentialDecay(self.params.critic_lr.base,
                                                                          decay_steps=self.params.critic_lr.decay_steps,
                                                                          decay_rate=self.params.critic_lr.decay_rate))

        self.discount_curve = DiscountCurve(self.params.gamma.base, rate=self.params.gamma.decay_rate,
                                            steps=self.params.gamma.decay_steps)

        self.num_actions = action_space.n
        render_shape = np.array((250, 750))
        if self.params.debug_observations:
            render_shape += np.array((300, 0))
        self.gym.register_render(self.render, shape=render_shape)
        self.rendering_params = copy.deepcopy(self.gym.params.rendering)
        self.gym.params.rendering.render = False

    def log_episode(self, info):
        self.logger.log_episode(info)

    def train(self):
        assert self.params.rollout_length % self.params.rollout_gyms == 0, "Cannot evenly balance rollout on gyms."
        training_steps = self.params.training_steps // self.params.rollout_gyms
        rollout_length = self.params.rollout_length // self.params.rollout_gyms

        gym_states = [self.gym.create_state() for _ in range(self.params.rollout_gyms)]
        states = [self.gym.reset(state=gym_state)[0] for gym_state in gym_states]
        self.gym.render(state=gym_states[0], params=self.rendering_params)
        state = {key: np.concatenate([s[key] for s in states], axis=0) for key in states[0].keys()}

        obs = self.observation_function(state)
        rollout = {
            "observations": {
                key: np.zeros([rollout_length, self.params.rollout_gyms] + value.shape[1:],
                              dtype=value.dtype.as_numpy_dtype) for key, value in obs.items()},
            "actions": np.zeros((rollout_length, self.params.rollout_gyms), dtype=np.int32),
            "probs": np.zeros((rollout_length, self.params.rollout_gyms), dtype=np.float32),
            "rewards": np.zeros((rollout_length, self.params.rollout_gyms), dtype=np.float32),
            "values": np.zeros((rollout_length, self.params.rollout_gyms), dtype=np.float32),
            "advantages": np.zeros((rollout_length, self.params.rollout_gyms), dtype=np.float32),
            "returns": np.zeros((rollout_length, self.params.rollout_gyms), dtype=np.float32)
        }
        episode_start = [0] * self.params.rollout_gyms
        rollout_infos = []

        progress = tqdm(total=self.params.training_steps)

        for step in range(training_steps):
            k = step % rollout_length  # Step in rollout

            action, prob, value = self.get_action_and_value(obs)

            next_states, reward, terminated, truncated, infos = zip(
                *[self.gym.step(state=gym_state, action=np.expand_dims(act, 0)) for gym_state, act in
                  zip(gym_states, action)])
            self.gym.render(state=gym_states[0], params=self.rendering_params)
            next_states = list(next_states)
            for info in infos:
                self.logger.log_step(info)

            for key, v in obs.items():
                rollout["observations"][key][k] = v
            rollout["actions"][k] = np.array(action)
            rollout["probs"][k] = np.array(prob)
            rollout["rewards"][k] = np.array(reward)
            rollout["values"][k] = np.array(value)

            rollout_full = k == rollout_length - 1

            for l in range(self.params.rollout_gyms):
                terminal = terminated[l]
                info = infos[l]
                if terminal or info["timeout"] or rollout_full:
                    # Episode done, fill advantages and returns
                    final_value = 0.0
                    if not terminal:
                        o = self.observation_function(next_states[l])
                        final_value = self.agent.get_value(o).numpy()[0]
                    self.fill_returns_and_advantages(rollout, episode_start[l], k, final_value, gym_id=l)
                    episode_start[l] = (k + 1) % rollout_length

                    if terminal or info["timeout"]:
                        self.log_episode(info)
                        rollout_infos.append(info)
                        state, _ = self.gym.reset(state=gym_states[l])
                        if l == 0:
                            self.gym.render(state=gym_states[0], params=self.rendering_params)
                        next_states[l] = state

            progress.update(self.params.rollout_gyms)

            if rollout_full:
                rollout_flat = {"observations": {}}
                for key, value in rollout.items():
                    if key == "observations":
                        for o, v in value.items():
                            rollout_flat["observations"][o] = np.reshape(v, [-1, *v.shape[2:]])
                    else:
                        rollout_flat[key] = np.reshape(value, (-1,))

                train_log = self.train_rollout(rollout_flat)
                self.logger.log_train(train_log)
                if len(rollout_infos) > 0:
                    self.logger.log_rollout(rollout_infos)
                rollout_infos = []

            state = {key: np.concatenate([s[key] for s in next_states], axis=0) for key in next_states[0].keys()}
            obs = self.observation_function(state)

    def get_action(self, obs, greedy=False):
        if greedy:
            action, prob = self.agent.get_exploitation_action(obs)
        else:
            action, prob = self.agent.get_exploration_action(obs)
        return action.numpy(), prob.numpy()

    def get_action_and_value(self, obs):
        action, probs, value = self.agent.get_action_prob_and_value(obs)
        return action.numpy(), probs.numpy(), value.numpy()

    def fill_returns_and_advantages(self, rollout, episode_start, episode_end, final_value, gym_id):
        rewards = rollout["rewards"][episode_start:episode_end + 1, gym_id]
        values = rollout["values"][episode_start:episode_end + 1, gym_id]

        advantage = self.compute_gae(rewards, values, final_value)
        rollout["advantages"][episode_start:episode_end + 1, gym_id] = advantage
        rollout["returns"][episode_start:episode_end + 1, gym_id] = advantage + values

    def compute_gae(self, rewards, values, final_value):
        """
        Compute the generalized advantage estimate.

        Parameters:
        - rewards: a list of rewards
        - values: a list of values

        Returns:
        - gae: the generalized advantage estimate
        """
        # Get the length of the rollout
        n = len(rewards)

        next_values = np.append(values[1:], final_value)
        # Load the discount factor and the GAE decay rate from the algorithm's parameters
        gamma = self.get_gamma()
        lam = self.params.lam

        # Compute the temporal difference (TD) errors as the difference between the rewards and the
        # values predicted by the critic network
        delta = rewards + gamma * next_values - values

        # Initialize the advantages to zero
        advantages = np.zeros(n + 1)

        # Iterate over the TD errors in reverse order
        for i in reversed(range(n)):
            # Compute the advantage for the current time step as an exponentially decaying sum of the TD errors
            advantages[i] = delta[i] + gamma * lam * advantages[i + 1]

        # Compute the generalized advantage estimate (GAE) as the sum of the advantages
        # and the difference between the values predicted by the critic network
        gae = advantages[:-1]

        # Return the computed GAE
        return gae

    def compute_returns(self, rewards, last_val):
        # Load the discount factor from the algorithm's parameters
        gamma = self.get_gamma()

        # Append the last value to the list of rewards
        returns = np.append(rewards, last_val)

        # Iterate over the rewards in reverse order
        for k in reversed(range(len(returns) - 1)):
            # Compute the return for the current time step as a sum of the reward and the discounted return at the
            # next time step
            returns[k] += gamma * returns[k + 1]

        # Return the computed returns, excluding the last value
        return returns[:-1]

    def train_epoch(self, dataset):
        logs = []
        for obs, actions, probs, advantages, returns in dataset.batch(self.params.batch_size):
            log = self.train_step_tf(
                obs=obs,
                actions=actions,
                old_probs=probs,
                advantages=advantages,
                returns=returns)
            logs.append(log)
        return logs

    @tf.function
    def train_step_tf(self, obs, actions, old_probs, advantages, returns):

        with tf.GradientTape() as critic_tape:
            value = self.agent.critic(obs)
            critic_loss = tf.keras.losses.Huber()(returns, tf.squeeze(value, -1))
        critic_grads = critic_tape.gradient(critic_loss, self.agent.critic.trainable_variables)
        critic_grads = [tf.clip_by_value(g, -1., 1.) for g in critic_grads]  # Clip gradients
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.agent.critic.trainable_variables))

        if self.params.normalize_advantage:
            advantages = (advantages - tf.reduce_mean(advantages, keepdims=True)) / (
                    tf.math.reduce_std(advantages, keepdims=True) + 1e-8)

        with tf.GradientTape() as tape:
            all_probs = self.agent.actor(obs)
            probs = tf.gather_nd(all_probs, tf.expand_dims(actions, -1), batch_dims=1)
            r = probs / old_probs
            epsilon = self.params.epsilon
            # Add the entropy term
            entropy = -tf.reduce_sum(all_probs * tf.math.log(all_probs + 1e-6), axis=-1)

            actor_loss = tf.minimum(r * advantages,
                                    tf.clip_by_value(r, 1. - epsilon, 1. + epsilon) * advantages
                                    ) + self.params.beta * entropy
            actor_loss = tf.where(tf.reduce_sum(tf.cast(obs["mask"], tf.int32), axis=-1) == 1, 0.0, actor_loss)

            actor_loss = -tf.reduce_mean(actor_loss)

        grads = tape.gradient(actor_loss, self.agent.actor.trainable_variables)
        grads = [tf.clip_by_value(g, -1., 1.) for g in grads]  # Clip gradients
        self.actor_optimizer.apply_gradients(zip(grads, self.agent.actor.trainable_variables))

        return {"critic_loss": critic_loss, "actor_loss": actor_loss, "entropy": tf.reduce_mean(entropy)}

    def rollout_to_dataset(self, rollout):
        obs = {key: tf.convert_to_tensor(value) for key, value in rollout["observations"].items()}
        actions = tf.convert_to_tensor(rollout["actions"], dtype=tf.int32)
        probs = tf.convert_to_tensor(rollout["probs"], dtype=tf.float32)
        advantages = tf.convert_to_tensor(rollout["advantages"], dtype=tf.float32)
        returns = tf.convert_to_tensor(rollout["returns"], dtype=tf.float32)

        dataset = tf.data.Dataset.from_tensor_slices((obs, actions, probs, advantages, returns))

        return dataset

    def train_rollout(self, rollout):
        dataset = self.rollout_to_dataset(rollout)
        logs = []
        for _ in range(self.params.rollout_epochs):
            dataset = dataset.shuffle(buffer_size=self.params.rollout_length)
            step_logs = self.train_epoch(dataset)
            log = dict_mean(step_logs)
            logs.append(log)
        return dict_mean(logs)

    def render(self, canvas_shape, state):
        canvas = pygame.Surface(canvas_shape, pygame.SRCALPHA, 32)
        canvas.fill((0, 0, 0))
        pygame.font.init()
        header_font = pygame.font.SysFont('Arial', 22, bold=True)
        font = pygame.font.SysFont('Arial', 16, bold=True)
        obs = self.observation_function(state)
        probs, value = self.agent.get_probs_and_value(obs)
        probs = probs.numpy()[0]
        value = value.numpy()[0]

        tile_size = canvas_shape[0] // 4
        draw_text(canvas, header_font, "Invariant action mask", (canvas_shape[0] / 2, 0), fill=(255, 255, 255),
                  align="top")
        mask_action_canvas = self.gym.draw_action_grid([f"{v:d}" for v in obs["mask"][0].numpy()], tile_size)
        canvas.blit(mask_action_canvas, (0, 30))

        draw_text(canvas, header_font, "pi(a|s)", (canvas_shape[0] / 2, tile_size * 4), fill=(255, 255, 255),
                  align="top")
        p_action_canvas = self.gym.draw_action_grid([f"{v * 100:.1f}" for v in probs], tile_size)
        canvas.blit(p_action_canvas, (0, tile_size * 4 + 30))

        draw_text(canvas, header_font, "V(s)", (canvas_shape[0] / 2, tile_size * 8), fill=(255, 255, 255), align="top")
        tile_canvas = self.gym.draw_tile(tile_size, "Value", f"{value[0]:.3f}")
        canvas.blit(tile_canvas, (int(tile_size * 1.5), tile_size * 8 + 30))

        offset = 250
        if self.params.debug_observations:
            global_map = obs["global_map"][0].numpy()
            local_map = obs["local_map"][0].numpy()

            obs_size = 100
            for k in range(global_map.shape[2]):
                global_layer = global_map[..., k]
                local_layer = local_map[..., k]

                image = cv2.resize(global_layer, dsize=[obs_size, obs_size], interpolation=cv2.INTER_NEAREST)
                image = np.repeat(np.expand_dims(image, -1), repeats=3, axis=-1) * 255
                surf = pygame.surfarray.make_surface(image.astype(np.uint8))
                canvas.blit(surf, (offset, (obs_size + 5) * k))
                pygame.draw.rect(canvas, (0, 0, 200), pygame.Rect((offset, (obs_size + 5) * k), (obs_size, obs_size)),
                                 width=1)

                image = cv2.resize(local_layer, dsize=[obs_size, obs_size], interpolation=cv2.INTER_NEAREST)
                image = np.repeat(np.expand_dims(image, -1), repeats=3, axis=-1) * 255
                surf = pygame.surfarray.make_surface(image.astype(np.uint8))
                canvas.blit(surf, (offset + 150, (obs_size + 5) * k))
                pygame.draw.rect(canvas, (0, 0, 200),
                                 pygame.Rect((offset + 150, (obs_size + 5) * k), (obs_size, obs_size)),
                                 width=1)

        return canvas

    def get_gamma(self):
        return self.discount_curve(self.logger.train_steps).numpy()
