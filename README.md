# Udacity RL nanodegree project 1

First project as part of the Udacity RL-course


## The Environment


For this project, student have to train an agent to navigate (and collect bananas!) in a large, square world. This is a custom Unity environment, created especially for Udacity course. More information about environment [here](https://github.com/jknthn/unity-banana-navigation)

![Image](https://github.com/alex-f1tor/udacity_rl_project_one/blob/master/imgs/environment_sample.png)

A **reward** of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The **state space** has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. 

Four discrete **actions** are available, corresponding to:

`0` - move forward.
`1` - move backward.
`2` - turn left.
`3` - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.
**
The following python3 libraries are required:
`numpy == 1.16.2`
`pytorch == 0.4.0` - (GPU enabled)
`unity ML-agent` - available at [github](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)



## DQN algorithims

As part of solving the problem, it was required to create an agent based on shallow neural network and train using the Q-learning approach.

For solving this problem, the following algorithms were implemented:

* Vanilla DQN [notebook](https://github.com/alex-f1tor/udacity_rl_project_one/blob/master/Vanilla%20DQN/Navigation%20Vanilla-DQN.ipynb)

* Double DQN; [notebook](https://github.com/alex-f1tor/udacity_rl_project_one/blob/master/Double%20DQN/Navigation-DDQN.ipynb)

* Dueling Neural Network;  [notebook](https://github.com/alex-f1tor/udacity_rl_project_one/blob/master/Dueling%20Neural%20Network/Navigation%20Dueling-DQN.ipynb)

* Prioritized Experience Replay; [notebook](https://github.com/alex-f1tor/udacity_rl_project_one/blob/master/Prioritized%20Experience%20Replay/Navigation%20Prioritized%20Experience%20Replay.ipynb)

* Noisy DQN: [notebook](https://github.com/alex-f1tor/udacity_rl_project_one/blob/master/Noisy%20DQN/Navigation-Noisy-DQN.ipynb)


## A little research

Despite the Atari results demonstrated at publications in the arxiv, it was interesting to know how various parameters will affect the training process of the agent in our current environment.

For each algorithm, the dependence of the duration of training on the its specific parameter, was explored. Since the score during learning process depends on several random factors, for each pair algorithm-parameter 3 neural networks was trained with the same set of parameters.

To create visual pretty graphics, 2 hints were used:
1) Smoothing with a moving average (window width 20) of the result in consecutive games (because original plots were noisy);
2) To show the variability, the minimum and maximum scores were selected for current episode number, which form a wide band (filled by color). The dashed line inside shows the median value for current episode number.


* **Vanilla DQN:** 

*Eps* - probability of random action instead of learning-based action, this value decay for each episode. 

![Image](https://github.com/alex-f1tor/udacity_rl_project_one/blob/master/imgs/vanilla_eps.png)

3 value ranges for Eps were explored: `1-0.01`, `1-0.5`, `0.5-0.01`. 
The **equally probable initial value** of Eps (Eps=0.5) allows to achieve faster convergence. It is very important to greatly **reduce the value at the end of the training**, otherwise the optimal solution will not be achieved .


* **Double DQN:**

*Tau* - how many trials\acts does the agent have between updating the weights of the target neural network;

![Image](https://github.com/alex-f1tor/udacity_rl_project_one/blob/master/imgs/ddqn_tau.png)

3 values for Tau were explored: `4`, `16`, `32`.
Comparing the results with the classical algorithm (the _green graph_ in the image of the previous section: all the double-networks in current section had the eps parameter ranged at [1-0.01]), a **significant acceleration** of convergence is observed for small time lag (Tau=4). The optimal solution will **not** be achieved in case of using a **big time lag**.


* **Dueling Neural Network:**

*Hidden layers* - are there additional hidden layer of a non-neural network in separated streams: advantage (A) and state value (V);

![Image](https://github.com/alex-f1tor/udacity_rl_project_one/blob/master/imgs/duel_hidden.png)

**No significant difference** is observed when using additional hidden layers at separated streams.


* **Prioritized Experience Replay**:

*Beta* - a schedule on the exponent  for annealing the amount of importance-sampling correction over time;

![Image](https://github.com/alex-f1tor/udacity_rl_project_one/blob/master/imgs/per_beta.png)

3 value ranges for Beta were explored: `0.1-1`, `0.5-1`, `0.8-1`. 
An arbitrary initial beta value leads to convergence of the algorithm. A **larger initial value** provides faster convergence.


* **Noisy DQN**:

*Sigma* - a standard deviation of distribution of noise implementing at each agent's act.

![Image](https://github.com/alex-f1tor/udacity_rl_project_one/blob/master/imgs/noisy_eps.png)

3 values for Tau were explored: `0.0017`, `0.017`, `0.17`.
Comparing the noisy-network's result with all other algorithms, the longest training procedure is observed.The optimal solution will **not** be achieved in case of using a **big amplitude of noise**.
