import numpy as np

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
        self.state_ = np.random.randint(0, self.maxcars+1, size=12)

    def return_state(self):
        non_zero = lambda x: x if x==0 else 1
        return [non_zero(line) for line in self.state_]

if __name__ == "__main__":
    env = Environment()
