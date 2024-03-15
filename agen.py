
from keras.models import load_model
import tensorflow as tf
import numpy as np
import random
from snack_ai import SnakeGameAI, Direction, Point, BLOCK_SIZE
from collections import deque
from keras.layers import Dense
import keras
from keras.models import Sequential
# from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam

from keras.metrics import mean_squared_error
MAX_MEMORY = 100_100
BATCH_SIZE = 100
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9  # count rate
        self.memory = deque(maxlen=MAX_MEMORY)
        # model
        self.model = Sequential()
        self.model.add(keras.Input((11,)))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(3, activation="linear"))
        optimizer = Adam(learning_rate=LR)
        self.model.compile(
            optimizer=optimizer, loss='mse', metrics=['accuracy'])
        self.model.load_weights("weight2.h5")

    def train_step(self, state, action, reward, next_state, done):

        # print(np.array(state))
        state = np.array(state, dtype='float64')
        action = np.array(action, dtype='int32')
        reward = np.array(reward, dtype='float64')
        next_state = np.array(next_state, dtype='float64')
        # done = np.array(done, dtype='float64')
        if len(state.shape) == 1:
            state = np.expand_dims(state, 0)
            action = np.expand_dims(action, 0)
            reward = np.expand_dims(reward, 0)
            next_state = np.expand_dims(next_state, 0)
            done = (done,)
        # predict Q with state curently

        # pred: np.ndarray = self.model.predict(np.array(state), verbose=0)
        # tagert = pred.copy()

        # Q_new=r+ y*max(Q(state))
        for idx in range(len(state)):
            Q_new = reward[idx]
            if not done[idx]:
                pred = self.model.predict(
                    np.array([next_state[idx]]), verbose=0)[0]
                Q_new = reward[idx] + self.gamma * np.max(pred)
            tagert: np.ndarray = self.model.predict(
                np.array([state[idx]]), verbose=0)[0]
            tagert[np.argmax(action[idx])] = Q_new
            self.model.fit(np.array([state[idx]]),
                           np.array([tagert]), verbose=0)

    def get_state(self, game: SnakeGameAI):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(
                self.memory, BATCH_SIZE)  # list or tuple
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # self.epsilon = 30-self.n_games
        self.epsilon = -1
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = np.array(state, dtype='float64')
            prediction = self.model.predict(np.array([state0]), verbose=0)[0]
            move = tf.math.argmax(prediction)
            final_move[move] = 1
        return final_move


def train():
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI(400, 400)
    while True:
        state_old = agent.get_state(game)

        final_move = agent.get_action(state_old)

        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        agent.train_short_memory(
            state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)
        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                record = score
                agent.model.save('weight.h5')
            # agent.model.save('weight.h5')

            print('Game: ', agent.n_games, "score: ", score, "record: ", record)


if __name__ == '__main__':
    train()

    pass
