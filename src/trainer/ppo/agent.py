from dataclasses import dataclass

from src.trainer.agent import Agent
from src.trainer.model import ModelFactory

import tensorflow as tf


class ACAgent(Agent):
    @dataclass
    class Params(Agent.Params):
        model: ModelFactory.default_param_type() = ModelFactory.default_params()

    def __init__(self, params, obs_space, act_space):
        super().__init__(params)
        self.actor = ModelFactory.create(params.model, obs_space=obs_space, act_space=act_space)
        self.critic = ModelFactory.create(params.model, obs_space=obs_space, act_space=None)

        self.expert_inference = False

    def save_network(self, path, name="model"):
        self.actor.model.save(f"{path}/actor_{name}")
        self.critic.model.save(f"{path}/critic_{name}")

    def load_network(self, path, name="model"):
        self.actor.model = tf.keras.models.load_model(f"{path}/actor_{name}")
        self.critic.model = tf.keras.models.load_model(f"{path}/critic_{name}")

    def load_weights(self, path, name="latest"):
        self.actor.model.load_weights(f"{path}/actor_{name}")
        self.critic.model.load_weights(f"{path}/critic_{name}")

    def save_weights(self, path, name="latest"):
        self.actor.model.save_weights(f"{path}/actor_{name}")
        self.critic.model.save_weights(f"{path}/critic_{name}")

    def save_keras(self, path):
        self.actor.model.save(f"{path}/actor.keras")
        self.critic.model.save(f"{path}/critic.keras")

    def load_keras(self, path):
        self.actor.model = tf.keras.models.load_model(f"{path}/actor.keras")
        self.critic.model = tf.keras.models.load_model(f"{path}/critic.keras")

    @tf.function
    def actor_inference(self, obs):
        return self.actor.predict_expert(obs) if self.expert_inference else self.actor(obs)

    @tf.function
    def critic_inference(self, obs):
        return self.critic.predict_expert(obs)[..., None] if self.expert_inference else self.critic(obs)

    @tf.function
    def get_probs_and_value(self, obs):
        probs = self.actor_inference(obs)
        value = self.critic_inference(obs)

        return probs, value

    @tf.function
    def get_action_prob_and_value(self, obs):
        action, probs = self.get_exploration_action(obs)
        value = self.get_value(obs)
        return action, probs, value

    @tf.function
    def get_value(self, obs):
        value = self.critic_inference(obs)
        return tf.squeeze(value, axis=-1)

    @tf.function
    def get_exploration_action(self, obs, step=None):
        probs = self.actor_inference(obs)
        actions = tf.random.categorical(tf.math.log(probs), 1)
        p = tf.gather_nd(probs, actions, batch_dims=1)
        actions = tf.squeeze(actions, axis=-1)
        return actions, p

    @tf.function
    def get_exploitation_action(self, obs):
        probs = self.actor_inference(obs)
        action = tf.argmax(probs, axis=-1)
        return action, tf.ones_like(action)

    def summary(self):
        print("Actor:")
        self.actor.model.summary()
        print("Critic:")
        self.critic.model.summary()
