"""
前提是動作空間離散且維度不高
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from maze_env import Maze

class Eval_Model(tf.keras.Model):


class Target_Model(tf.keras.Model):


class DQN:
    def __init__(self, n_actions, n_features, eval_model, target_model):
        self.params = {
            'n_actions': n_actions,
            'n_features': n_features,
            'learning_rate': 0.01,
            'reward_decay': 0.9,
            'e_greedy': 0.9,
            'replace_target_iter': 300,
            'memory_size': 500,
            'batch_size': 32,
            'e_greedy_increment': None
        }
