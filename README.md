# TrafficLights
This is a small project I used to get a first taste of reinforcement learning. The goal of the project is to teach a traffic light system to quickly clear intersections. The environment is a four-way intersection, with three lanes in each direction: one turning left, one going straight ahead and the last turning right. In order to model the intersection being observed by simple sensors, the agent only sees the first car in each lane. The agent then sets the trafficlights (takes an action) allowing some cars to pass the intersection. Once one wave of cars has gone through, the agent gets another opportunity to change the state of the trafficlights (take next action). The process continues until the intersection is cleared or a maximum number of actions has been taken.

# Quickstart
The file Qlearn.py features classes for the environment (intersection) and the agent. First, an agent must be initialized:
```
model = Agent()
```
The agent can then be trained using the .train() method, or read in from a file using the .load_model() method. After training, the agent can be given a new environment with the .refresh_environment() method, refilling the intersection with cars. Then, the .solve() method can be used to let the agent attempt to clear the intersection. The method returns a boolean flag signaling whether the agent was successful and the number of actions taken. Passing the verbose=True flag to the .solve() method prints a visualisation of the intersection to the console. For example, one could train and run an agent like this:
```
model.train(episodes=1000)
model.refresh_environment()
success, actions = model.solve(verbose=True)
```
To save an agent as a file, use the .save_model() method.

To evaluate the agent's efficiency, the .test_model() method can be used. It creates a given amount of test cases and lets the agent attempt to solve them. It returns the amount of successful test cases and the average efficiency of the agent. The efficiency of the agent on a given test case is defined as the ratio between the optimal amount of actions and the actual amount of actions taken. Thus, in an intersection which could optimall be cleared in 3 actions but the agent took 4, the efficiency is 0.75.
