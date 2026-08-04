Yatzy – Project Specification

Programming Language
- Python

Program:
-Bachelor of Computer science (TKT)

(I can review projects in Python.)

Algorithms and Data Structures
- Q-learning network (Multi Layer Perceptron)
Algorithms: forward pass, backward pass, stochastic gradient descent

Data structures:
- Arrays/lists/matrices for dice, scoretable and network weights

Problem Being Solved:
-Teaching an AI agent to play yatzy using Deep-Q-learning network

Program Inputs:
-Interface is the yatzy game code, which will give the state and action for the Q-learning agent to choose from. 
The user can change the hyperparameters of the network such as epsilon, epsilon decay, gamma, learning rate, etc.

Program outputs:
The program outputs training progress every N games (episodes) to see what the average score is that the agent achieves with current level of training

For example: Episode 100 | Avg Score: 49.92 | Bonus %: 0.00 | Epsilon: 0.904

--> The agent has played 100 games, In the last 100 games the average score it achieved in a game was 49.92, it didnt recieve the upper bonus in any of the games 0% and
the epsilon parameter which dictates to what degree the agent explores new actions vs exploits best currently known action for the given state. This will decay over training
to make the agent choose actions that it has learned that lead to a high score.

Time and Space Complexity
For forward pass and backward pass the time complexity is 
O(∑n_(i)​ * n_(i+1)​) where n_i is the size of the i:th layer.
For this project the network has size Input: 21 --> Hidden 1: 128 --> Hidden 2: 128 --> Output: 37
O(21*128+128*128+128*37) = O(23808)


Space complexity:
O(P+A)
where P are the parameters and A is number of activations. For this network its 
P = 21*128+128*128+128*37 = 23808
A = 21+128+128+37=314

Sources
Listed in the resources_used.txt file in the documentation folder.

Core of the Project
The main focus of the project is implementing the Q-learning algorithm to play the yatzy game, and to implement the game interface. If there
is more time left, output/visualization and UI will be improved.