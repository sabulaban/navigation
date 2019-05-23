


# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  



A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Navigation.ipynb` to get started with training your own agent!  


The repo consists of 5 folders:
- classes: This is where the agent and the pytorch models are defined, agent.py consists of the following classes:
    - Vanilla DQN: Simple implementation of DQN
    - Duel Agent: DQN with Duelling using the QNetwork_Duel class in model.py
    - Duel Double Agent: DQN implementation with Duelling and Doubling where Q_next is selected based on A_next which follows the same Epsilon/Greedy policy used for Q_current
    - Duel Double PER Agent: Prioritized Experience Replay implementation of A Duelling DDQN
- workspace: A collection of notebooks capturing the training for the above agents
    - Vanilla DQN
    - Duelling DQN
    - Duelling DDQN
    - Duelling DDQN + PER (file name is: Navigation.ipynb)
    - Agents Comparison: A comparison in terms of average score over 100 episodes per agent 
- pickle: 1 pickle file per agent, with the score per agent is retained
- model: a folder used to store the best performing model per agent
- logdir: a folder (not used in code) to dump logs if needed

### File: workspace/Navigation.ipynb

Training and presentation of results for Duelling DDQN agent with PER

### File: workspace/agents.comparison.ipynb

Comprehensive comparison between different agents performance during training, and playing the game


```python

```
