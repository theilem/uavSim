from dataclasses import dataclass
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, Dense, Concatenate, MaxPool2D, ReLU
from tensorflow.keras import Model

from utils import Factory


class GlobLocModel:
    @dataclass
    class Params:
        # Convolutional part config
        conv_layers: int = 4
        conv_kernel_size: int = 3
        conv_kernels: int = 32

        # Fully Connected config
        hidden_layer_size: int = 256
        hidden_layer_num: int = 3

    def __init__(self, params: Params, obs_space, act_space=None):
        self.params = params
        self.observation_space = obs_space
        self.action_space = act_space
        self.model = self.create_model()

    def __call__(self, obs):
        return self.model(obs)

    @property
    def trainable_variables(self):
        return self.model.trainable_variables

    def create_model(self):
        obs = self.observation_space
        global_map_input = Input(shape=obs["global_map"].shape[1:], dtype=tf.float32)
        local_map_input = Input(shape=obs["local_map"].shape[1:], dtype=tf.float32)
        scalars_input = Input(shape=obs["scalars"].shape[1:], dtype=tf.float32)
        mask_input = Input(shape=obs["mask"].shape[1:], dtype=tf.bool)

        global_map = global_map_input
        local_map = local_map_input
        conv_kernels = self.params.conv_kernels
        hidden_layer_size = self.params.hidden_layer_size
        kernel_size = self.params.conv_kernel_size

        global_map = Conv2D(conv_kernels, 1, activation=None)(global_map)  # linear pixel-wise embedding
        local_map = Conv2D(conv_kernels, 1, activation=None)(local_map)  # linear pixel-wise embedding

        # Feature Extraction
        for _ in range(self.params.conv_layers):
            global_map = Conv2D(conv_kernels, kernel_size, activation="relu", padding="same")(global_map)
            local_map = Conv2D(conv_kernels, kernel_size, activation="relu", padding="same")(local_map)

            conv_kernels *= 2
            global_map = Conv2D(conv_kernels, kernel_size, activation="relu", padding="same")(global_map)
            local_map = Conv2D(conv_kernels, kernel_size, activation="relu", padding="same")(local_map)
            global_map = MaxPool2D(2)(global_map)
            local_map = MaxPool2D(2)(local_map)

        global_features = tf.reduce_max(tf.reduce_max(global_map, axis=1), axis=1)
        local_features = tf.reduce_max(tf.reduce_max(local_map, axis=1), axis=1)

        layer = tf.concat((local_features, global_features, scalars_input), axis=1)

        hidden_layer_num = self.params.hidden_layer_num
        for k in range(hidden_layer_num):
            layer = Dense(hidden_layer_size, activation="relu")(layer)

        if self.action_space is None:
            # Value head
            outputs = Dense(1, activation=None)(layer)
        else:
            layer = Dense(self.action_space.n, activation=None)(layer)
            layer = tf.where(mask_input, layer, -np.inf)
            outputs = tf.math.softmax(layer, axis=-1)

        return Model(
            inputs={"global_map": global_map_input, "local_map": local_map_input, "scalars": scalars_input,
                    "mask": mask_input},
            outputs=outputs)


class ModelFactory(Factory):
    @classmethod
    def registry(cls):
        return {
            "glob_loc": GlobLocModel
        }
