import numpy as np

class Environment:
    def __init__(self, maxcars = 3):
        self.maxcars = maxcars
        self.state_ = np.random.randint(0, self.maxcars+1, size=12)

    def return_state(self):
        non_zero = lambda x: x if x==0 else 1
        return [non_zero(line) for line in self.state_]

if __name__ == "__main__":
    test = Environment()
    print(test.state_)
    print(test.return_state())