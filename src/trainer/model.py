from dataclasses import dataclass
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPool2D, Flatten, Embedding
from tensorflow.keras import Model

from utils import Factory


class NNModel:

    def __init__(self, params, obs_space, act_space=None):
        self.params = params
        self.observation_space = obs_space
        self.action_space = act_space
        self.model = self.create_model()

    @tf.function
    def __call__(self, obs):
        return self.model(obs)

    def add_head(self, layer, mask):
        if self.action_space is None:
            # Value head
            outputs = Dense(1, activation=None)(layer)
        else:
            layer = Dense(self.action_space.n, activation=None)(layer)
            layer = tf.where(mask, layer, -np.inf)
            outputs = tf.math.softmax(layer, axis=-1)
        return outputs

    @property
    def trainable_variables(self):
        return self.model.trainable_variables

    def create_model(self):
        raise NotImplementedError("Implement in subclass")


class MapModel(NNModel):
    @dataclass
    class Params:
        # Convolutional part config
        conv_layers: int = 4
        conv_kernel_size: int = 3
        conv_kernels: int = 32

        # Fully Connected config
        hidden_layer_size: int = 256
        hidden_layer_num: int = 3

        # Conv to FC conversion
        use_pooling: bool = True
        conversion: str = "reduce"  # Supported "reduce" and "flatten"

    def __init__(self, params: Params, obs_space, act_space=None):
        super().__init__(params, obs_space, act_space)

    def create_model(self):
        obs = self.observation_space
        map_input = Input(shape=obs["map"].shape[1:], dtype=tf.float32)
        scalars_input = Input(shape=obs["scalars"].shape[1:], dtype=tf.float32)
        mask_input = Input(shape=obs["mask"].shape[1:], dtype=tf.bool)

        plain_map = map_input
        conv_kernels = self.params.conv_kernels
        hidden_layer_size = self.params.hidden_layer_size
        kernel_size = self.params.conv_kernel_size

        plain_map = Conv2D(conv_kernels, 1, activation=None)(plain_map)  # linear pixel-wise embedding

        # Feature Extraction
        for _ in range(self.params.conv_layers):
            plain_map = Conv2D(conv_kernels, kernel_size, activation="relu", padding="same")(plain_map)

            conv_kernels *= 2
            plain_map = Conv2D(conv_kernels, kernel_size, activation="relu", padding="same")(plain_map)

            if self.params.use_pooling:
                plain_map = MaxPool2D(2)(plain_map)

        if self.params.conversion == "flatten":
            features = Flatten()(plain_map)
        elif self.params.conversion == "reduce":
            features = tf.reduce_max(tf.reduce_max(plain_map, axis=1), axis=1)
        else:
            raise NotImplementedError(f"Unknown conversion: {self.params.conversion}")

        layer = tf.concat((features, scalars_input), axis=1)

        hidden_layer_num = self.params.hidden_layer_num
        for k in range(hidden_layer_num):
            layer = Dense(hidden_layer_size, activation="relu")(layer)

        outputs = self.add_head(layer, mask_input)

        return Model(
            inputs={"map": map_input, "scalars": scalars_input, "mask": mask_input},
            outputs=outputs)


class GlobLocModel(NNModel):
    @dataclass
    class Params:
        # Convolutional part config
        conv_layers: int = 4
        conv_kernel_size: int = 3
        conv_kernels: int = 32

        # Fully Connected config
        hidden_layer_size: int = 256
        hidden_layer_num: int = 3

        # Conv to FC conversion
        use_pooling: bool = True
        conversion: str = "reduce"  # Supported "reduce" and "flatten"

    def __init__(self, params: Params, obs_space, act_space=None):
        super().__init__(params, obs_space, act_space)

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
            if self.params.use_pooling:
                global_map = MaxPool2D(2)(global_map)
                local_map = MaxPool2D(2)(local_map)

        if self.params.conversion == "flatten":
            global_features = Flatten()(global_map)
            local_features = Flatten()(local_map)
        elif self.params.conversion == "reduce":
            global_features = tf.reduce_max(tf.reduce_max(global_map, axis=1), axis=1)
            local_features = tf.reduce_max(tf.reduce_max(local_map, axis=1), axis=1)
        else:
            raise NotImplementedError(f"Unknown conversion: {self.params.conversion}")

        layer = tf.concat((local_features, global_features, scalars_input), axis=1)

        hidden_layer_num = self.params.hidden_layer_num
        for k in range(hidden_layer_num):
            layer = Dense(hidden_layer_size, activation="relu")(layer)

        outputs = self.add_head(layer, mask_input)

        return Model(inputs={"global_map": global_map_input, "local_map": local_map_input, "scalars": scalars_input,
                             "mask": mask_input}, outputs=outputs)


class RotEquivarianceModel(MapModel):
    @dataclass
    class Params(MapModel.Params):
        pass

    def __init__(self, params: Params, obs_space, act_space=None):
        super().__init__(params, obs_space, act_space)
        if act_space is not None:
            assert act_space.n == 7, "Symmetry hard-coded for 7 actions"
        self.params = params

    def create_model(self):
        obs = self.observation_space
        map_input = Input(shape=obs["map"].shape[1:], dtype=tf.float32)
        scalars_input = Input(shape=obs["scalars"].shape[1:], dtype=tf.float32)

        plain_map = map_input
        conv_kernels = self.params.conv_kernels
        hidden_layer_size = self.params.hidden_layer_size
        kernel_size = self.params.conv_kernel_size

        plain_map = Conv2D(conv_kernels, 1, activation=None)(plain_map)  # linear pixel-wise embedding

        # Feature Extraction
        for _ in range(self.params.conv_layers):
            plain_map = Conv2D(conv_kernels, kernel_size, activation="relu", padding="same")(plain_map)

            conv_kernels *= 2
            plain_map = Conv2D(conv_kernels, kernel_size, activation="relu", padding="same")(plain_map)

            if self.params.use_pooling:
                plain_map = MaxPool2D(2)(plain_map)

        if self.params.conversion == "flatten":
            features = Flatten()(plain_map)
        elif self.params.conversion == "reduce":
            features = tf.reduce_max(tf.reduce_max(plain_map, axis=1), axis=1)
        else:
            raise NotImplementedError(f"Unknown conversion: {self.params.conversion}")

        layer = tf.concat((features, scalars_input), axis=1)

        hidden_layer_num = self.params.hidden_layer_num
        for k in range(hidden_layer_num):
            layer = Dense(hidden_layer_size, activation="relu")(layer)

        outputs = self.add_head(layer, None)

        return Model(
            inputs={"map": map_input, "scalars": scalars_input},
            outputs=outputs)

    def add_head(self, layer, mask):
        if self.action_space is None:
            # Value head
            outputs = Dense(1, activation=None)(layer)
        else:
            outputs = Dense(self.action_space.n, activation=None)(layer)
        return outputs

    @tf.function(jit_compile=True)
    def __call__(self, obs):
        if self.action_space is None:
            # Critic
            return self.model(obs)
        # Actor, apply mask
        mask = obs["mask"]
        outputs = self.model(obs)
        masked = tf.where(mask, outputs, -np.inf)
        return tf.math.softmax(masked, axis=-1)

    @tf.function(jit_compile=True)
    def predict_equiv(self, obs):
        scalars = obs["scalars"]
        mask = obs["mask"]
        map_tensor = obs["map"]

        map_tensor_90 = tf.image.rot90(map_tensor, k=1)
        map_tensor_180 = tf.image.rot90(map_tensor, k=2)
        map_tensor_270 = tf.image.rot90(map_tensor, k=3)

        stacked = tf.stack((map_tensor, map_tensor_90, map_tensor_180, map_tensor_270), axis=1)  # [b, 4, h, w, c]

        scalars_repeated = tf.repeat(scalars, repeats=4, axis=0)

        b, _, h, w, c = stacked.shape
        reshaped = tf.reshape(stacked, (-1, h, w, c))

        rotated_obs = {"map": reshaped, "scalars": scalars_repeated}

        output = self.model(rotated_obs)

        if self.action_space is None:
            # Critic
            values = tf.reshape(output, (-1, 4))
            return values

        # Actor
        action_names = {
            0: "right",
            1: "down",
            2: "left",
            3: "up",
            4: "land",
            5: "take off",
            6: "charge"
        }

        actions = tf.reshape(output, (-1, 4, 7))
        dir_actions, non_dir_actions = tf.split(actions, [4, 3], axis=2)

        actions0, actions90, actions180, actions270 = tf.unstack(dir_actions, axis=1)
        actions90 = tf.roll(actions90, -1, axis=-1)
        actions180 = tf.roll(actions180, -2, axis=-1)
        actions270 = tf.roll(actions270, -3, axis=-1)

        dir_actions_corrected = tf.stack((actions0, actions90, actions180, actions270), axis=1)
        actions_corrected = tf.concat((dir_actions_corrected, non_dir_actions), axis=2)  # [b, 4, 7]

        actions_masked = tf.where(mask[:, None, :], actions_corrected, -np.inf)
        soft = tf.math.softmax(actions_masked, axis=2)

        return soft

    @tf.function(jit_compile=True)
    def predict_expert(self, obs):
        values = self.predict_equiv(obs)
        return tf.reduce_mean(values, axis=1)


class ModelFactory(Factory):
    @classmethod
    def registry(cls):
        return {
            "glob_loc": GlobLocModel,
            "map": MapModel,
            "equiv": RotEquivarianceModel
        }
