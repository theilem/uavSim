import os
from collections import deque
from dataclasses import dataclass
from typing import Tuple, List, Optional

import tensorflow as tf

from utils import dict_mean


class Logger:
    @dataclass
    class Params:
        loss_period: int = 100
        evaluation_period: int = 10_000
        evaluation_start: int = 25_000
        evaluation_episodes: int = 1
        save_weights: int = 100_000
        save_keras: int = 1_000_000
        log_episodes: bool = True
        save_specific: Optional[List[int]] = None

    def __init__(self, params: Params, log_dir, agent):
        self.params = params
        self.log_dir = log_dir
        self.evaluator = None
        self.agent = agent
        self.agent.save_network(f"{self.log_dir}/models/")
        self.agent.save_weights(f"{self.log_dir}/models/")

        self.log_writer = tf.summary.create_file_writer(self.log_dir + '/training')
        print(f"Logging to: \n{os.getcwd()}/{self.log_dir}")

        self.train_steps = 0
        self.steps = 0

        self.train_logs = deque(maxlen=self.params.loss_period)
        self.action_probs = []

    def log_train(self, train_log):
        self.train_steps += 1

        if self.params.loss_period == 1:
            logs = train_log
        else:
            self.train_logs.append(train_log)
            if self.train_steps % self.params.loss_period != 0:
                return
            logs = dict_mean(self.train_logs)
            
        with self.log_writer.as_default():
            for name, value in logs.items():
                self.log_scalar(f'training/{name}', value, self.train_steps)

    def log_step(self, step_info, action_prob=None):
        self.steps += 1
        if action_prob is not None:
            self.action_probs.append(action_prob)

        if self.steps % self.params.evaluation_period == 0 and self.steps >= self.params.evaluation_start:
            if self.evaluator is not None:
                if self.params.evaluation_episodes == 1:
                    info = self.evaluator.evaluate_episode()
                else:
                    infos = [self.evaluator.evaluate_episode() for _ in range(self.params.evaluation_episodes)]
                    info = dict_mean(infos)
                with self.log_writer.as_default():
                    for name, value in info.items():
                        self.log_scalar(f'evaluation/{name}', value, self.steps)

        if self.params.save_specific is not None:
            for step in self.params.save_specific:
                if self.steps % step == 0:
                    self.agent.save_weights(f"{self.log_dir}/models/", name=f"weights_at_{step}")


        if self.steps % self.params.save_weights == 0:
            self.agent.save_weights(f"{self.log_dir}/models/")
        if self.steps % self.params.save_keras == 0:
            self.agent.save_keras(f"{self.log_dir}/models/")

    def log_episode(self, info):
        if not self.params.log_episodes:
            return

        with self.log_writer.as_default():
            for name, value in info.items():
                self.log_scalar(f'episodic/{name}', value, self.steps)

    def log_rollout(self, infos):
        with self.log_writer.as_default():
            mean_info = dict_mean(infos)
            for name, value in mean_info.items():
                self.log_scalar(f'rollout/{name}', value, self.steps)

    def log_scalar(self, name, value, step):
        if isinstance(value, str):
            return
        tf.summary.scalar(name, value, step)
