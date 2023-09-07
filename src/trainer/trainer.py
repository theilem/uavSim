from dataclasses import dataclass
from typing import Tuple

from gymnasium import spaces

from src.gym import GridGym
import tensorflow as tf

from src.trainer.utils import DecayParams
from utils import Factory
from tensorflow.keras.optimizers.schedules import ExponentialDecay


def polynomial_activation(layer, degree=3):
    return tf.concat([layer ** d for d in range(1, degree + 1)], axis=-1)


class DiscountCurve:
    def __init__(self, base, rate, steps):
        self.decay = ExponentialDecay(1 - base, decay_rate=rate, decay_steps=steps)
        self.use_decay = rate != 1.0
        self.base = tf.constant(base)

    def __call__(self, step):
        if self.use_decay:
            return 1.0 - self.decay(step)
        else:
            return self.base


class BaseTrainer:
    @dataclass
    class Params:
        training_steps: int = 2_000_000
        gamma: DecayParams = DecayParams(0.96, 0.5, 1_000_000)

    def __init__(self, gym: GridGym, observation_function, action_space):
        self.gym = gym
        self.observation_function = observation_function

        assert isinstance(action_space, spaces.Discrete)
        self.action_space: spaces.Discrete = action_space
        self.observation_space = observation_function.get_observation_space(gym.observation_space.sample())

    def get_action(self, obs, greedy=False) -> Tuple[int, float]:
        raise NotImplementedError()

    @staticmethod
    @tf.function
    def _soft_update_tf(network, target_network, alpha):
        weights = network.weights
        target_weights = target_network.weights
        new_weights = [w_new * alpha + w_old * (1. - alpha) for w_new, w_old
                       in zip(weights, target_weights)]
        [target_weight.assign(new_weight) for new_weight, target_weight in zip(new_weights, target_weights)]


class TrainerFactory(Factory):

    @classmethod
    def registry(cls):
        from src.trainer.ppo.ppo import PPOTrainer
        return {
            "ppo": PPOTrainer
        }
