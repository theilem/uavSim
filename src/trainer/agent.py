from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from utils import Factory


class Agent:
    @dataclass
    class Params:
        pass

    def __init__(self, params):
        self.params = params

    def save_network(self, path, name="model"):
        raise NotImplementedError()

    def save_weights(self, path, name="weights_latest"):
        raise NotImplementedError()

    def load_network(self, path, name="model"):
        raise NotImplementedError()

    def load_weights(self, path, name="weights_latest"):
        raise NotImplementedError()

    def save_keras(self, path):
        raise NotImplementedError()

    def load_keras(self, path):
        raise NotImplementedError()

    def summary(self):
        pass

    @tf.function
    def get_random_action(self, obs):
        mask = obs["mask"]
        return tf.random.categorical(tf.where(mask, 1., -np.inf), 1)[..., 0], 1.0 / tf.reduce_sum(
            tf.cast(mask, tf.float32), axis=-1)

    @tf.function
    def get_exploration_action(self, obs, step):
        raise NotImplementedError()

    @tf.function
    def get_exploitation_action(self, obs):
        raise NotImplementedError()


class AgentFactory(Factory):

    @classmethod
    def registry(cls):
        from src.trainer.ppo.agent import ACAgent

        return {
            "ac": ACAgent
        }
