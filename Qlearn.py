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
    """Class representing an agent, who changes the trafficlight state to clear cars waiting at an intersection. Mechanically, takes actions
    given in Environment.actions on an Environment instance to clear its state.
    
    attributes
    ----------
    env : Environment
        the Environment instance the agent operates on.

    Q : np.array
        the Q-table which governs its decisions.

    methods
    -------
    state_to_index_(state)
        converts an Environment state to its corresponding index in the Q-table.

    epsilon_greedy_policy(epsilon)
        decides whether the agent should act randomly (exploration) or as learned from experience (exploitation).

    train(episodes, max_eps=1.0, min_eps=0.05, learning_rate=0.7, decay_rate=0.0005, gamma=0.95, max_steps=10)
        trains the model by letting it act on a state, rewarding it upon success and modifying its Q-table.

    refresh_environment()
        generates a new random state for the agent's environment.

    experience_stats()
        returns the indices of the agent's Q-table where not all entires are zero, meaning indices where the agent has learned something.

    solve(max_steps = 10, verbose = False)
        attempts to solve the current state of the agent's environment by applying what it has learned (exploitation).

    test_model(testnum, max_complex = 5, max_steps = 10)
        lets the model run through a given amount of test cases to evaluate its performance.

    save_model(filepath)
        saves the agent's Q-table to a file.

    load_model(filepath)
        loads the agent's Q-table from a file.
    """

    def __init__(self, environment = Environment()):
        """
        parameters
        ----------
        environment : Environment
            The environment instance the agent will operate on.
        """
        self.env = environment
        self.Q = np.zeros((2**self.env.lanes, len(self.env.actions.keys())))  #Q table

    def state_to_index_(self, state):
        '''converts a given environment state to its corresponding index in the Q table.
        
        parameters
        ----------
        state : np.array
            the state array (as given by Environment.return_state() ) to look up.
            
        returns
        -------
        res : int
            the index in the agent's Q-table, such that the relevant row can be found in self.Q[res].
        '''
        res = 0
        for i in range(len(state)):
            res += state[::-1][i]*2**i
        return res
    
    def epsilon_greedy_policy(self, epsilon):
        '''Decies whether the agent should act randomly (exploration) or based on its learned behaviour (exploitation) and
        returns the corresponding action key.
        
        parameters
        ----------
        epsilon : float
            The chance for which the agent acts randomly. Should be between 0 and 1.
            
        returns
        -------
        action key : string
            name of the action the agent will take, as called in Environment.actions.
        '''
        random_num = random.uniform(0,1)
        if random_num < epsilon:
            return random.choice(list(self.env.actions.keys()))
        else:
            index = np.argmax(self.Q[self.state_to_index_(self.env.return_state())])
            return list(self.env.actions.keys())[index]
        
    def train(self, episodes, max_eps=1.0, min_eps=0.05, learning_rate=0.7, decay_rate=0.0005, gamma=0.7, max_steps=10):
        '''trains the model by letting it act on a state, rewarding it upon success and modifying its Q-table.
        
        parameters
        ----------
        episodes : int
            the number of training episodes the agent goes through.

        max_eps : float, optional
            The maximum epsilon value during training, determining the chance for random behaviour. Standard is 1.0.

        min_eps : float, optional
            The minimum epsilon value during training, determining the chance for random behaviour. Standard is 0.05.
        
        learning_rate : float, optional
            The rate at which training influences the already learned behaviour. A higher value means a higher willingness to learn.

        decay_rate : float, optional
            The rate at which the epsilon value decays over training. Standard value is 0.0005.

        gamma : float, optional
            The factor with which later rewards are weighted compared to immediate ones. Standard value is 0.7.

        max_steps : int, optional
            The maximum amount of actions the agent is allowed to take during training. Standard value is 10.
        '''
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
        '''generates a new random state for the agent's environment.'''
        self.env = Environment()
                

    def experience_stats(self):
        '''returns the cases where the agent has some idea what to do, i.e. the indices of the Q-table where not all values are zero.
        
        returns
        -------
        indices : np.array
            an array of indices where the current Q-table does not only hold zero-values.
        '''
        indices = [i for i in range(np.shape(self.Q)[0]) if np.any(self.Q[i]) != 0]
        return indices

    def solve(self, max_steps = 10, verbose = False):
        '''attempts to solve the current state of the agent's environment by applying what it has learned (exploitation).
        
        parameters
        ----------
        max_steps : int, optional
            The maximum amount of actions the agent is allowed to take before the attempt is considered a failure.
            Standard value is 10.
        
        verbose : bool, optional
            Flag determining whether the steps the agent takes should be represented visually.
            If set to True, updates in the agent's solution attempt are printed to console.
            Standard is False.

        returns
        -------
        success : bool
            Whether the agent succeeded in clearing the state.

        steps : int
            The amount of actions the agent took in the solution attempt.
        '''
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
        '''lets the model run through a given amount of test cases to evaluate its performance.
        
        parameters
        ----------
        testnum : int
            The  number of test cases the agent is put through.

        max_complex : int, optional
            The maximum complexity for test cases, i.e. how many actions are at least
            necessary to solve the test cases. Standard value is 5.
        
        max_steps : int, optional
            Amount of actions the agent is allowed to take on each test case before the attempt
            is considered a failure. Standard value is 10.

        returns
        -------
        success_count : int
            The amount of successfully solved test cases.

        efficiency : float
            The average efficiency over all test cases. The efficiency is defined as the ratio between
            optimal number of actions to solve the test case and the needed steps. If the agent did
            not succeed in solving the test case, the efficiency is considered to be 0. For example,
            a test case with complexity 4, which was solved in 5 action would yield an efficiency
            of 0.8.
        '''
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
        '''saves the Q-table of this Agent to a file located in filepath.
        parameters
        ----------
        filepath : string
            The filepath to the file where the Q-table should be stored.
        '''
        np.savetxt(filepath, self.Q, delimiter=',')

    def load_model(self, filepath):
        '''loads the Q-table from a given file in location filepath.
        
        parameters
        ----------
        filepath : string
            The filepath where the agent's Q-table is stored.
        '''
        self.Q = np.loadtxt(filepath, delimiter=",")