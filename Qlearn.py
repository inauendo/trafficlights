import numpy as np
import random
from print_util import *

class Environment:
    """Class representing a four-way intersection, used as environment for the agent.
    
    attributes
    ----------
    actions : dict
        a dictionary containing all possible trafficlight settings, i.e. the actions the agent can take.
        Each action is represented by an array of length 12, where 1 means the car can drive, 0 means it must wait.
        Lanes are labeled counter-clockwise, beginning at the bottom (south).

    maxcars : int
        The maximum amount of cars waiting in each lane.

    lanes : int
        The amount of lanes at the intersection.

    state_ : np.array[int]
        Numpy array representing the amount of cars in each lane. Lanes are labelled counter-clockwise, starting at the bottom (south).

    actions_taken : int
        How many actions have already been performed, i.e. how many waves of cars have passed through the intersection.
    
    methods
    -------
    return_state()
        returns a numpy array with size lanes that carries a 1 for each lane where at least one car is waiting, and 0 otherwise.

    perform_action(action_key)
        performs a given action on the intersection.

    rotate_state_(state)
        rotates the given state 90° counter-clockwise.

    find_permutations_()
        finds all possible intersection states by rotating the current intersection's state

    generate_test_case(complexity)
        generates an intersection state which can be resolved in a given number of actions, given by complexity, and sets the
        current state to this.

    print_state(action_key = "")
        prints a visualisation of the current state and, if given, the planned action given by action_key to the console.

    """

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
        """Constructs a random state for the intersection.
        
        parameters
        ----------
        maxcars : int, optional
            The maximum amount of cars in each lane. Default is 3.
        """
        self.maxcars = maxcars
        self.lanes = 12
        self.state_ = np.random.randint(0, self.maxcars+1, size=self.lanes)
        self.actions_taken = 0

    def return_state(self):
        '''Returns the current state as seen by the agent, i.e. whether cars are present in lanes or not.
        
        returns
        -------
        state : np.array
            A numpy array of length Environment.lanes, where a 1 denotes a lane occupied by a waiting car, and 0 denotes a free lane.
        '''
        non_zero = lambda x: x if x==0 else 1
        return np.array([non_zero(line) for line in self.state_])
    
    def perform_action(self, action_key):
        '''Performs a given action on the current state, returning the new state and its reward, if any.
        
        parameters
        ----------
        action_key : string
            the name of the action as called in Environment.actions.

        returns
        -------
        new_state : np.array
            the new state of the intersection, after the actions has been taken.

        reward : int
            the reward granted if the action managed to clear the intersection.

        raises
        ------
        ValueError
            If action_key is not a valid key for Environment.actions.
        '''
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
        '''finds all non-trivial states that can be found by rotating the given state.
        
        parameters
        ----------
        state : np.array
            array of representing the intersection state.
            
        returns
        -------
        res : np.array
            the same state, rotated by 90° counter-clockwise.
        '''
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
    
    def generate_test_case(self, complexity):
        '''sets the current state to a random state where the optimal amount of steps necessary to solve it equals the given complexity.
        
        parameters
        ----------
        complexity : int
            the optimal amount of actions necessary to clear the intersection.
        '''
        self.state_ = np.zeros_like(self.state_)
        for i in range(complexity):
            self.state_ += random.choice(list(self.actions.values()))

    def print_state(self, action_key = ""):
        '''prints a visualisation of the current intersection state and planned action, if given, to the console.
        
        parameters
        ----------
        action_key : string, optional
            name of the planned action, as called in Environment.actions. If not given (standard), no action is printed.
        '''
        _ = os.system("")
        if action_key == "":
            action = np.zeros_like(self.state_)
        else:
            action = self.actions[action_key]
        print(state_string(self.state_, action))

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
            self.env = Environment(maxcars=1)
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

            if episode % (episodes//20) == 0:
                barlength = episode//(episodes//20)
                print((barlength*'\u2588' + (20-barlength)*'-'+" {ratio:.0%}  training...").format(ratio=episode/episodes), end='\r')
            
        print((20*'\u2588' + " {ratio:.0%}  training finished.").format(ratio=1))

    def refresh_environment(self):
        '''refreshes the environment.'''
        self.env = Environment()
                

    def experience_stats(self):
        '''returns the cases where the agent has some idea what to do, i.e. the row numbers of the Q-table where not all values are zero.'''
        indices = [i for i in range(np.shape(self.Q)[0]) if np.any(self.Q[i]) != 0]
        return indices

    def solve(self, max_steps = 10, verbose = False):
        '''tries to apply the current Q-table to solve the environment.'''
        reward = 0
        steps = 0
        success = True
        while reward == 0:
            action_key = list(self.env.actions.keys())[np.argmax(self.Q[self.state_to_index_(self.env.return_state())])]
            if verbose == True:
                _ = os.system('cls')
                print("Current state:")
                self.env.print_state()
                _ = input("\n\nPress any key to continue.")
                _ = os.system('cls')
                print("Action:")
                self.env.print_state(action_key=action_key)
                _ = input("\n\nPress any key to continue.")
            new_state, reward = self.env.perform_action(action_key)

            steps += 1
            if steps >= max_steps:
                if verbose == True:
                    print("Maximum number of steps reached.")
                success = False
                break
        
        if reward != 0 and verbose == True:
            _ = os.system('cls')
            print("Current state:")
            self.env.print_state()

        return success, steps

    def test_model(self, testnum, max_complex = 5, max_steps = 10):
        '''tests the model, where testnum denotes the amount of test runs and max_complex the maximal amount of necessary actions to clear the environment in the optimal case.
        Max_steps is the same flag as in solve.
        Returns the amount of successful test cases and the average efficiency.'''
        success_count = 0
        efficiency_sum = 0
        for i in range(testnum):
            complexity = random.choice(np.arange(1, max_complex+1))     #randomly set complexity of the test run
            self.env.generate_test_case(complexity=complexity)
            flag, steps = self.solve(max_steps=max_steps)
            if flag == True:
                success_count += 1
                efficiency_sum += complexity/steps
        return success_count, efficiency_sum/testnum
    
    def save_model(self, filepath):
        '''saves the Q-table of this Agent to a file located in filepath.'''
        np.savetxt(filepath, self.Q, delimiter=',')

    def load_model(self, filepath):
        '''loads the Q-table from a given file in location filepath.'''
        self.Q = np.loadtxt(filepath, delimiter=",")