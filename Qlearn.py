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

        reward = 1/self.actions_taken
        for lane in self.state_:
            if lane != 0:
                reward = 0  #reset reward if there are still non-zero lanes

        new_state = self.return_state()
        return new_state, reward
    
    def rotate_state_(self, state):
        return np.concatenate((state[-3:], state[:-3]))
    
    def find_permutations_(self, state):
        '''finds an returns all non-trivial states that can be found by rotating the given state.'''
        res = [state]
        tmpstate = state
        for i in range(3):
            tmpstate = self.rotate_state_(tmpstate)
            not_present = True
            for entry in res:
                if np.all(entry == tmpstate):
                    not_present = False 
            if not_present == True:
                res.append(tmpstate)
        return np.array(res)

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
        
    def train(self, episodes, max_eps=1.0, min_eps=0.05, learning_rate=0.7, decay_rate=0.0005, gamma=0.95, max_steps=10):
        '''trains the model, modifying the Q-table.'''
        for episode in range(episodes):
            self.env = Environment()
            steps = 0
            eps = min_eps + (max_eps - min_eps)*np.exp(-decay_rate*episode)

            for step in range(max_steps):
                action_key = self.epsilon_greedy_policy(eps)
                state = self.env.return_state()
                new_state, reward = self.env.perform_action(action_key)

                #edit Q-table for every valid rotation of state and action
                action_tuples = [(state, action_key, new_state)]
                tmpstate = state
                tmpactionkey = action_key
                tmpnewstate = new_state
                for i in range(3):
                    tmpstate = self.env.rotate_state_(tmpstate)
                    tmpactionkey = list(self.env.actions.keys())[list(self.env.actions.values()).index(list(self.env.rotate_state_(self.env.actions[tmpactionkey])))]
                    tmpnewstate = self.env.rotate_state_(tmpnewstate)
                    not_present = True
                    for entry in action_tuples:
                        if np.all(entry[0] == tmpstate):
                            not_present = False
                            break
                    if not_present == True:
                        action_tuples.append((tmpstate, tmpactionkey, tmpnewstate))

                for (state, action_key, new_state) in action_tuples:
                    action_index = np.argwhere(np.array(list(self.env.actions.keys())) == action_key)[0][0]
                    stateindex = self.state_to_index_(state)
                    self.Q[stateindex][action_index] = self.Q[stateindex][action_index] + learning_rate * (reward + gamma * np.max(self.Q[self.state_to_index_(new_state)]) - self.Q[stateindex][action_index])

                if reward != 0:
                    break

            if (episode+1) % int(episodes/20) == 0:
                print(int(episode/episodes * 100), "% done...")

    def experience_stats(self):
        '''returns the cases where the agent has some idea what to do, i.e. the row numbers of the Q-table where not all values are zero.'''
        indices = [i for i in range(np.shape(self.Q)[0]) if np.any(self.Q[i]) != 0]
        return indices

    def solve(self, max_steps = 10):
        '''tries to apply the current Q-table to solve the environment.'''
        reward = 0
        steps = 0
        print("Current state:", self.env.return_state())
        while reward == 0:
            action_key = list(self.env.actions.keys())[np.argmax(self.Q[self.state_to_index_(self.env.return_state())])]
            print("applying action:", action_key)
            new_state, reward = self.env.perform_action(action_key)
            print("current state:", new_state)

            steps += 1
            if steps >= max_steps:
                print("Maximum number of steps reached.")
                break

if __name__ == "__main__":
    env = Environment()
    ag = Agent()
