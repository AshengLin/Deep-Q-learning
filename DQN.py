"""
前提是動作空間離散且維度不高
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from maze_env import Maze

class Eval_Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_q_network')
        self.layer1 = layers.Dense(10, activation='relu')
        self.logits = layers.Dense(num_actions, activation=None)

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        layer1 = self.layer1(x)
        logits = self.logits(layer1)
        return logits

class Target_Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_q_network_1')
        self.layer1 = layers.Dense(10, trainable=False, activation='relu')
        self.logits = layers.Dense(num_actions, trainable=False, activation=None)

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        layer1 = self.layer1(x)
        logits = self.logits(layer1)
        return logits

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
        self.learn_step_counter = 0
        self.epsilon = 0 if self.params['e_greedy_increment'] is not None else self.params['e_greedy']
        self.memory = np.zeros((self.params['memory_size'], self.params['n_features'] * 2 + 2))

        self.eval_model = eval_model
        self.target_model = target_model

        self.eval_model.compile(
            optimizer=tf.keras.optimizers.RMSprop(lr=self.params['learning_rate']),
            loss='mse'
        )
        self.cost_his = []

    def choose_action(self, observation):


    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


def run_maze():  # 與環境交互
    step = 0
    for episode in range(300):
        observation = env.reset()  # reset
        while True:
            env.render()  # 刷新環境10次


if __name__ == "__main__":
    env = Maze()
    eval_model = Eval_Model(num_actions=env.n_actions)
    target_model = Target_Model(num_actions=env.n_actions)
    RL_model = DQN(env.n_actions, env.n_features, eval_model, target_model)
    env.after(100, run_maze)
    env.mainloop()
    RL_model.plot_cost()





