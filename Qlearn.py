import numpy as np
import random

class Environment:

    actions = {
        'all_right': [0,0,1,0,0,1,0,0,1,0,0,1],
        'down_straight': [0,1,1,0,0,0,0,0,1,0,0,1],
        'right_straight': [0,0,1,0,1,1,0,0,0,0,0,1],
        'up_straight': [0,0,1,0,0,1,0,1,1,0,0,0],
        'left_straight': [0,0,0,0,0,1,0,0,1,0,1,1],
        'up_down_straight': [0,1,1,0,0,0,0,1,1,0,0,0],
        'left_right_straight': [0,0,0,0,1,1,0,0,0,0,1,1],
        'down_all': [1,1,1,0,0,0,0,0,0,0,0,1],
        'right_all': [0,0,1,1,1,1,0,0,0,0,0,0],
        'up_all': [0,0,0,0,0,1,1,1,1,0,0,0],
        'left_all': [0,0,0,0,0,0,0,0,1,1,1,1],
        'down_left': [1,0,1,0,0,1,0,0,0,0,0,1],
        'right_left': [0,0,1,1,0,1,0,0,1,0,0,0],
        'up_left': [0,0,0,0,0,1,1,0,1,0,0,1],
        'left_left': [0,0,1,0,0,0,0,0,1,1,0,1],
        'up_down_left': [1,0,0,0,0,1,1,0,0,0,0,1],
        'left_right_left': [0,0,1,1,0,0,0,0,1,1,0,0]
        }

    def __init__(self, maxcars = 3):
        self.maxcars = maxcars
        self.lanes = 12
        self.state_ = np.random.randint(0, self.maxcars+1, size=self.lanes)
        self.actions_taken = 0

    def return_state(self):
        '''returns the state an agent sees.'''
        non_zero = lambda x: x if x==0 else 1
        return np.array([non_zero(line) for line in self.state_])
    
    def perform_action(self, action_key):
        '''performs action corresponding to action_key on the state, returns new state and reward.'''
        if action_key not in self.actions.keys():
            raise ValueError('Passed action key does not correspond to a valid action!')
        
        temp = self.state_ - self.actions[action_key]
        self.state_ = np.where(temp >= 0, temp, 0)
        self.actions_taken += 1

        reward = -self.actions_taken
        for lane in self.state_:
            if lane != 0:
                reward = 0  #reset reward if there are still non-zero lanes

        new_state = self.return_state()
        return new_state, reward

class Agent:
    def __init__(self, environment = Environment()):
        self.env = environment
        self.Q = np.zeros((2**self.env.lanes, len(self.env.actions.keys())))  #Q table

    def state_to_index_(self, state):
        '''converts a lane state (as given by self.return_state) to the corresponding index in the Q table.'''
        res = 0
        for i in range(len(state)):
            res += state[::-1][i]*2**i
        return res
    
    def epsilon_greedy_policy(self, epsilon):
        '''returns the action key corresponding to the action to take determined by the epsilon greedy policy.'''
        random_num = random.uniform(0,1)
        if random_num < epsilon:
            return random.choice(list(self.env.actions.keys()))
        else:
            index = np.argmax(self.Q[self.state_to_index_(self.env.return_state())])
            return list(self.env.actions.keys())[index]
        
    def train(self, episodes, max_eps, min_eps, learning_rate, decay_rate, gamma, max_steps):
        '''trains the model, modifying the Q-table.'''
        for episode in range(episodes):
            self.env = Environment()
            steps = 0
            eps = min_eps + (max_eps - min_eps)*np.exp(-decay_rate*episode)

            for step in range(max_steps):
                action_key = self.epsilon_greedy_policy(eps)
                state = self.env.return_state()
                new_state, reward = self.env.perform_action(action_key)

                action_index = np.argwhere(list(self.env.actions.keys()) == action_key)[0][0]
                self.Q[self.state_to_index_(state)][action_index] = self.Q[self.state_to_index_(state)][action_index] + learning_rate * (reward + gamma * np.max(self.Q[self.state_to_index_(new_state)]) - self.Q[self.state_to_index_(state)][action_index])

                if reward != 0:
                    break

if __name__ == "__main__":
    env = Environment()
    ag = Agent()
