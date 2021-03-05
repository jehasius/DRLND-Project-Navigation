# Navigation Project - Udacity Deep Reinforcement Learning Expert Nanodegree
#### J. Hampe 01/2021.
---

[1. Introduction](#intro)  
[2. Getting Started](#start)  
[3. Learning Algorithm](#algo)  
[4. Plot of Rewards](#plot)  
[5. Simulation](#sim)  
[6. Ideas for future work](#future)  

[//]: # (Image References)
[image1]: ./pictures/score_episode.png "Score over Episode"
[image2]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

<a name="intro"></a>
## 1. Introduction
This project was a part of the Udacity Deep Reinforcement Learning Expert Nanodegree course. The main task of this project, called Navigation, was to use Deep Q-Learning to train a machine learning agent to collect yellow bananas in an Unity environment. To find out more about the used Unity environment please look at [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents) explainations. The Deep Q-Learning algorithm is an algorithm of the Reinforment Learning class, in  which an optimal action-value function is represented as a neural network (instead of a table).

### 1.2 AI - Machine Learning - Reinforment Learning
Machine learning, is a branch of artificial intelligence, focuses on learning patterns from data. The three main classes of machine learning algorithms include: unsupervised learning, supervised learning and reinforcement learning. Each class of algorithm learns from a different type of data. You can find a good overview over the main concepts of machine learning at [Unity Background-Machine-Learning](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Background-Machine-Learning.md)

<a name="start"></a>
## 2. Getting Started
To set up and run the environment please follow the generell instructions in the [README.md](./README.md) and Jupyter notebook [Navigation.ipynb](./Navigation.ipynb). The Jupyter notebook als contains the whole project algorithms.

<a name="algo"></a>
## 3. Learning Algoritm
The used Deep Q-Learning algorithm is an algorithm of the Reinforment Learning class, in  which an optimal action-value function is represented as a neural network (instead of a table). To find out more abaot the algrithm please consider the research paper that introduces the Deep Q-Learning algorithm: [Human-level control through deep reinforcementlearning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf).

### 3.1. Network Architecture
I this project I uesed a simple neural network architecture to represent the optimal action-value function as you can see below.

```python
QNetwork(
  (fc1): Linear(in_features=37, out_features=64, bias=True)
  (fc1_drop): Dropout(p=0.2, inplace=False)
  (fc2): Linear(in_features=64, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=64, bias=True)
  (fc4): Linear(in_features=64, out_features=4, bias=True)
```
### 3.2 DRL Hyperparameter
To tune the a Deep Reinforcement Learning system you are always have a bunch of hyperprameter in your algoritm. With this hyperparameters you can influence and optimize the hole learnig prozess.

```python
BUFFER_SIZE = int(1e5)	# replay buffer size
BATCH_SIZE = 64           # minibatch size
GAMMA = 0.99              # discount factor
TAU = 1e-3                # for soft update of target parameters
LR = 5e-4                 # learning rate
UPDATE_EVERY = 4          # how often to update the network
eps_start (float)         : starting value of epsilon, for epsilon-greedy action selection
eps_end (float)           : minimum value of epsilon
eps_decay (float)         : multiplicative factor (per episode) for decreasing epsilon
```
### 3.3 Build and train the Deep Q-Learning algorithm
The following steps were used to build and train the Deep Q-Learning algorithm:

1. Creating a network model QNetwork
2. Creating a ML-Agent
3. Setting up the agent
4. Train the Agent with DQN and save the trained model parameter the neural network
5. Load the model parameter in the neural network and test the agent (train_mode=False)

<a name="plot"></a>
## 3. Plot of Rewards
The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.Therefore a reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas. The Environment was solved in 660 episodes!	with an average score: 15.02 yellow bananas.
```python
Episode 100	Average Score: 0.79
Episode 200	Average Score: 3.75
Episode 300	Average Score: 7.69
Episode 400	Average Score: 11.02
Episode 500	Average Score: 12.85
Episode 600	Average Score: 13.95
Episode 700	Average Score: 14.07
Episode 760	Average Score: 15.02
Environment solved in 660 episodes!	Average Score: 15.02
```
In the picture below you can see the approproate plot of the rewards during the training process as score over episodes.

![alt text][image1]  

## 4. Simulation<a name="sim"></a>
The model was successfully trained and the agent was able to collect the desired amound of yellow bananas.

![alt text][image2]  

<a name="future"></a>
## 5. Ideas for future work
The alorithm is running quit well but we can try to tune hyper-parameters.
Testing different network architecture that usees other layers like convolutions or use a deeper network architectures.
Testing different variations of the learning algorithm like:
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- [Dueling DQN](https://arxiv.org/abs/1511.06581)
- [Rainbow](https://arxiv.org/abs/1710.02298)

