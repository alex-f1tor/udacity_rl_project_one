# Udacity RL nanodegree project 1

First project as part of the Udacity RL-course


## The Environment


For this project, student have to train an agent to navigate (and collect bananas!) in a large, square world. This is a custom Unity environment, created especially for Udacity course. More information here: ![Link]

A **reward** of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The **state space** has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. 

Four discrete **actions** are available, corresponding to:

`0` - move forward.
`1` - move backward.
`2` - turn left.
`3` - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.


## DQN algorithims

As part of solving the problem, it was required to create an agent based on shallow neural network and train using the Q-learning approach.

For solving this problem, the following algorithms were implemented:

* Vanilla DQN [application](https://www.google.com), [model file](https://www.google.com); 

* Double DQN; 

* Dueling Neural Network; 

* Prioritized Experience Replay; 

* Noisy DQN:


## A little research

Despite the Atari results demonstrated at publications in the arxiv, it was interesting to know how various parameters will affect the training process of the agent in our current environment.

For each algorithm, the dependence of the duration of training on the its specific parameter, was explored:

* Vanilla DQN: 

*Eps* - probability of random action instead of learning-based action;

* Double DQN: 

*Tau* - how many trials\acts does the agent have between updating the weights of the target neural network;

* Dueling Neural Network:

*Hidden layers* - are there additional hidden layer of a non-neural network in separated streams: advantage (A) and state value (V);

* Prioritized Experience Replay:

*Beta* - a schedule on the exponent  for annealing the amount of importance-sampling correction over time;

* Noisy DQN:

*Sigma* - a standard deviation of distribution of noise implementing at each agent's act.
