The general structure of the program:

The project has three folders:

1. Algorithms

This folder has the algorithms that play the yatzy game.

q_learning.py: This was the Q-learning table algorithm that was replaced by the neural network so it is no longer used.

neural_net.py This is the file that implements the Deep-Q-Network and all the necessary algorithms for it like gradient descent, forward and backward pass.

replaybuffer.py This is related to the Deep-Q-Network, it is the buffer that saves the games played by the agent. It specifically saves the (state, action, reward, next_state)
                tuples the agent plays. The Network is then trained on a random batch of these experiences on every round of the game.

2. Game

This folder has all the necessary code for the yatzy game engine.

dice.py: Includes class that has functionality for the dice that are used in the yatzy game

scoreboard.py: Includes class that implements the scoreboard for the yatzy game

yatzy.py: Implements the yatzy game and uses dice.py and scoreboard.py

3. Training

This folder has the implementation for the training loop, it combines the AI agent and the game engine

training.py: Here we setup the training loop and also print out the training progress.
             It is modular by design so you could theoretically change the game and agent, and the training loop wouldnt need to be changed.


--------------------------------------------------------------------

Achieved time and space complexities (e.g., Big-O analysis based on pseudocode):
-

--------------------------------------------------------------------

Performance and Big-O complexity comparison (if relevant to the project):
-

--------------------------------------------------------------------
Possible shortcomings and suggestions for improvement:

From week6-report:

- I have tried all possible changes and combinations of changes from activation clipping, network weight clipping, Q-value clipping, changing learning parameters, implementing replay buffer, implementing target network, changing rewards and their sizes but the neural network is still struggling to learn. Without replay buffer the network scored average score of around 85 per game which is sub optimal and might be due to some learned reward hacking instead of solid strategy. The learning also later collapsed back to an average score of 50 indicating unstable learning. Next improvement trials would be to try to change the state representation of the game to give the agent more awareness of the game like scores in each category instead of only 0/1 depending on if that category has been used and then run training for long period of time on a cluster. Before that, some profiling and code optimization would also be beneficial to achieve faster training. 

--------------------------------------------------------------------

Use of large language models (ChatGPT, etc.). Mention which model you used and how. If you did not use any, explicitly state that. This is important!:

I used LLMs to help brainstorm ideas like repository folder structure and also to trouble shoot reasons
why the agent wasn't learning and what factors could affect that.

--------------------------------------------------------------------

List of the sources you have used, only those relevant to your work.

NeuralNine - Q-Learning Tutorial in Python - Reinforcement Learning
https://www.youtube.com/watch?v=MSrfaI1gGjI

Piotr Skalski - Let’s code a Neural Network in plain NumPy
https://medium.com/data-science/lets-code-a-neural-network-in-plain-numpy-ae7e74410795

NumPy community - Deep learning on MNIST
https://numpy.org/numpy-tutorials/tutorial-deep-learning-on-mnist/

Anyscale - Practical tips for training Deep Q Networks By Misha Laskin
https://www.anyscale.com/blog/practical-tips-for-training-deep-q-networks

Testing Neural Networks – A blog post by Sebastian Björkqvist.
https://www.sebastianbjorkqvist.com/blog/writing-automated-tests-for-neural-networks/