import copy
from dataclasses import dataclass

import cv2
import numpy as np
import pygame
from tqdm import tqdm

import tensorflow as tf

from src.gym.utils import draw_text
from src.trainer.trainer import BaseTrainer, DiscountCurve, DiscountSuccessSchedule
from src.trainer.utils import DecayParams, dict_to_tensor, dict_slice, dict_slice_set

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
        use_success_discount: bool = False

        debug_observations: bool = False

        regularize: bool = False
        use_expert: bool = False

    def __init__(self, params: Params, gym, agent, logger):
        super().__init__(gym)
        self.params = params
        self.agent = agent
        self.logger = logger

        self.agent.expert_inference = self.params.use_expert

        # Compute optimizer steps per interaction step
        opt_steps = self.params.rollout_epochs / self.params.batch_size

        actor_decay = ExponentialDecay(self.params.actor_lr.base,
                                       decay_steps=self.params.actor_lr.decay_steps * opt_steps,
                                       decay_rate=self.params.actor_lr.decay_rate)
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_decay)
        critic_decay = ExponentialDecay(self.params.critic_lr.base,
                                        decay_steps=self.params.critic_lr.decay_steps * opt_steps,
                                        decay_rate=self.params.critic_lr.decay_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_decay)

        if self.params.use_success_discount:
            self.discount_curve = DiscountSuccessSchedule(self.params.gamma.base, rate=self.params.gamma.decay_rate,
                                                          steps=self.params.gamma.decay_steps)
        else:
            self.discount_curve = DiscountCurve(self.params.gamma.base, rate=self.params.gamma.decay_rate,
                                                steps=self.params.gamma.decay_steps)

        self.num_actions = gym.action_space.n
        render_shape = np.array((250, 750))
        if self.params.debug_observations:
            render_shape += np.array((300, 0))
        self.gym.register_render(self.render, shape=render_shape)
        self.rendering_params = copy.deepcopy(self.gym.params.rendering)
        self.gym.params.rendering.render = False
        self.training_logs = {"critic_loss": tf.Variable(0.0),
                              "actor_loss": tf.Variable(0.0),
                              "entropy": tf.Variable(0.0)}

    def train(self):
        assert self.params.rollout_length % self.params.rollout_gyms == 0, "Cannot evenly balance rollout on gyms."
        training_steps = self.params.training_steps // self.params.rollout_gyms
        rollout_length = self.params.rollout_length // self.params.rollout_gyms

        gym_states = [self.gym.create_state() for _ in range(self.params.rollout_gyms)]
        observes = [self.gym.reset(state=gym_state)[0] for gym_state in gym_states]
        self.gym.render(state=gym_states[0], params=self.rendering_params)
        obs = {key: np.concatenate([o[key] for o in observes], axis=0) for key in observes[0].keys()}

        rollout = {
            "observations": {
                key: np.zeros([rollout_length, self.params.rollout_gyms] + list(value.shape[1:]),
                              dtype=value.dtype) for key, value in obs.items()},
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

            next_observes, reward, terminated, truncated, infos = self.gym.step_multi(gym_states, action)
            self.gym.render(state=gym_states[0], params=self.rendering_params)
            for info in infos:
                self.logger.log_step(info)

            for key, v in obs.items():
                rollout["observations"][key][k] = v
            rollout["actions"][k] = np.array(action)
            rollout["probs"][k] = np.array(prob)
            rollout["rewards"][k] = np.array(reward)
            rollout["values"][k] = np.array(value)

            rollout_full = k == (rollout_length - 1)

            for l in range(self.params.rollout_gyms):
                terminal = terminated[l]
                trunc = truncated[l]
                info = infos[l]
                if terminal or trunc or rollout_full:
                    # Episode done, fill advantages and returns
                    final_value = 0.0
                    if not terminal:
                        o = dict_slice(next_observes, l)
                        final_value = self.agent.get_value(o).numpy()[0]
                    self.fill_returns_and_advantages(rollout, episode_start[l], k, final_value, gym_id=l)
                    episode_start[l] = (k + 1) % rollout_length

                    if terminal or trunc:
                        if info["task_solved"]:
                            self.discount_curve.log_success()
                        self.logger.log_episode(info)
                        rollout_infos.append(info)
                        o, _ = self.gym.reset(state=gym_states[l])
                        if l == 0:
                            self.gym.render(state=gym_states[0], params=self.rendering_params)
                        dict_slice_set(next_observes, l, o)

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

            obs = next_observes

    def get_action(self, obs, greedy=False):
        if greedy:
            action, prob = self.agent.get_exploitation_action(obs)
        else:
            action, prob = self.agent.get_exploration_action(obs)
        return action.numpy(), prob.numpy()

    def get_action_and_value(self, obs):
        action, probs, value = self.agent.get_action_prob_and_value(dict_to_tensor(obs))
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

    @tf.function
    def get_grads_tf(self, obs, actions, old_probs, advantages, returns):
        with tf.GradientTape() as critic_tape:
            critic_loss = tf.constant(0.0, dtype=tf.float32)
            if self.params.regularize:
                values = self.agent.critic.predict_equiv(obs)
                mean_value = tf.reduce_mean(values, axis=1, keepdims=True)
                critic_loss += tf.reduce_mean((values - tf.stop_gradient(mean_value)) ** 2)
                value = mean_value if self.params.use_expert else values[..., 0, None]
            else:
                value = self.agent.critic_inference(obs)
            critic_loss += tf.keras.losses.Huber()(returns, tf.squeeze(value, -1))
        critic_grads = critic_tape.gradient(critic_loss, self.agent.critic.trainable_variables)

        with tf.GradientTape() as tape:
            actor_loss = tf.constant(0.0, dtype=tf.float32)
            if self.params.regularize:
                all_probs = self.agent.actor.predict_equiv(obs)  # [B, 4, A]

                target = tf.stop_gradient(tf.reduce_mean(all_probs, axis=1, keepdims=True))
                klr = tf.reduce_sum(all_probs * (tf.math.log(all_probs + 1e-6) - tf.math.log(target + 1e-6)),
                                    axis=-1)
                klf = tf.reduce_sum(target * (tf.math.log(target + 1e-6) - tf.math.log(all_probs + 1e-6)), axis=-1)
                kl = klr + klf
                actor_loss += tf.reduce_mean(kl)

                all_probs = tf.reduce_mean(all_probs, axis=1) if self.params.use_expert else all_probs[:, 0]
            else:
                all_probs = self.agent.actor_inference(obs)
            probs = tf.gather_nd(all_probs, tf.expand_dims(actions, -1), batch_dims=1)
            r = probs / old_probs
            epsilon = self.params.epsilon
            # Add the entropy term
            entropy = -tf.reduce_sum(all_probs * tf.math.log(all_probs + 1e-6), axis=-1)

            loss = tf.minimum(r * advantages,
                              tf.clip_by_value(r, 1. - epsilon, 1. + epsilon) * advantages
                              ) + self.params.beta * entropy
            loss = tf.where(tf.reduce_sum(tf.cast(obs["mask"], tf.int32), axis=-1) == 1, 0.0, loss)

            actor_loss -= tf.reduce_mean(loss)

        grads = tape.gradient(actor_loss, self.agent.actor.trainable_variables)

        return {"critic_loss": critic_loss, "actor_loss": actor_loss, "entropy": tf.reduce_mean(entropy),
                "actor_grads": grads, "critic_grads": critic_grads}

    @tf.function
    def train_step_tf(self, obs, actions, old_probs, advantages, returns):

        values = self.get_grads_tf(obs, actions, old_probs, advantages, returns)
        critic_grads = values.pop("critic_grads")
        actor_grads = values.pop("actor_grads")

        critic_grads = [tf.clip_by_value(g, -1., 1.) for g in critic_grads]  # Clip gradients
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.agent.critic.trainable_variables))

        actor_grads = [tf.clip_by_value(g, -1., 1.) for g in actor_grads]  # Clip gradients
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.agent.actor.trainable_variables))

        return values

    def train_rollout(self, rollout):
        obs = {key: tf.convert_to_tensor(value) for key, value in rollout["observations"].items()}
        actions = tf.convert_to_tensor(rollout["actions"], dtype=tf.int32)
        probs = tf.convert_to_tensor(rollout["probs"], dtype=tf.float32)
        advantages = tf.convert_to_tensor(rollout["advantages"], dtype=tf.float32)
        returns = tf.convert_to_tensor(rollout["returns"], dtype=tf.float32)

        random_idx = np.concatenate(
            [np.random.permutation(self.params.rollout_length) for _ in range(self.params.rollout_epochs)],
            axis=0)
        # Cut off remainder
        remainder = (random_idx.shape[0] % self.params.batch_size)
        if remainder > 0:
            random_idx = random_idx[:-remainder]
        # Reshape to have batch definitions
        random_idx = np.reshape(random_idx, (-1, self.params.batch_size))

        random_idx = tf.convert_to_tensor(random_idx, dtype=tf.int32)

        self.train_batches_tf(obs=obs, actions=actions, probs=probs, advantages=advantages, returns=returns,
                              random_idx=random_idx)

        logs = {"gamma": self.get_gamma()}
        logs.update(self.training_logs)

        return logs

    @tf.function
    def train_batches_tf(self, obs, actions, probs, advantages, returns, random_idx):
        num_batches = random_idx.shape[0]
        for value in self.training_logs.values():
            value.assign(0.0)

        if self.params.normalize_advantage:
            advantages = (advantages - tf.reduce_mean(advantages, keepdims=True)) / (
                    tf.math.reduce_std(advantages, keepdims=True) + 1e-8)

        for batch_idx in random_idx:
            obs_batch = {key: tf.gather(value, batch_idx) for key, value in obs.items()}
            actions_batch = tf.gather(actions, batch_idx)
            probs_batch = tf.gather(probs, batch_idx)
            advantages_batch = tf.gather(advantages, batch_idx)
            returns_batch = tf.gather(returns, batch_idx)

            batch_logs = self.train_step_tf(obs_batch, actions_batch, probs_batch, advantages_batch, returns_batch)
            for key, value in self.training_logs.items():
                value.assign_add(batch_logs[key] / num_batches)

    def render(self, canvas_shape, obs):
        canvas = pygame.Surface(canvas_shape, pygame.SRCALPHA, 32)
        canvas.fill((0, 0, 0))
        pygame.font.init()
        header_font = pygame.font.SysFont('Arial', 22, bold=True)
        probs, value = self.agent.get_probs_and_value(obs)
        probs = probs.numpy()[0]
        value = value.numpy()[0]

        tile_size = canvas_shape[0] // 4
        draw_text(canvas, header_font, "Invariant action mask", (canvas_shape[0] / 2, 0), fill=(255, 255, 255),
                  align="top")
        mask_action_canvas = self.gym.draw_action_grid([f"{v:d}" for v in obs["mask"][0]], tile_size)
        canvas.blit(mask_action_canvas, (0, 30))

        draw_text(canvas, header_font, "pi(a|s)", (canvas_shape[0] / 2, tile_size * 4), fill=(255, 255, 255),
                  align="top")
        p_action_canvas = self.gym.draw_action_grid([f"{v * 100:.1f}" for v in probs], tile_size)
        canvas.blit(p_action_canvas, (0, tile_size * 4 + 30))

        draw_text(canvas, header_font, "V(s)", (canvas_shape[0] / 2, tile_size * 8), fill=(255, 255, 255),
                  align="top")
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
                pygame.draw.rect(canvas, (0, 0, 200),
                                 pygame.Rect((offset, (obs_size + 5) * k), (obs_size, obs_size)),
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
        return self.discount_curve(self.logger.train_steps * self.params.rollout_length).numpy()
