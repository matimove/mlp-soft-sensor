import random
import math
import numpy as np
import itertools
import copy

class NN:
    """Class that implements a feed forward neural network (MLP)

    Attributes:
        net: Stores the neural nets weights
        gamma: Discount factor for future rewards
        lr: Learning rate for the neural network
        epsilon: Probability of choosing a random action over an optimal action
        epsilon_decay: Factor that shrinks the epsilon after each game
        min_epsilon: Value that the epsilon wont go lower than
        actions: List of available actions for the agent to take
        categories: List of all of the combos in yatzy
        action_to_index: Map between actions and their index in the neural network outputs
        counter: Used to freeze a copy of the net for N rounds.
        target_net: Copy of the neural net for N rounds to help stabilize training.
    """

    def __init__(self):
        self.net = {}
        self.gamma = 0.99
        self.lr = 1e-3
        self.epsilon = 1
        self.epsilon_decay = 0.999
        self.min_epsilon = 0.05
        self.actions = {}
        self.categories = [
            "ones", "twos", "threes", "fours", "fives", "sixes",
            "one_pair", "two_pair", "three_of_a_kind", "four_of_a_kind",
            "small_straight", "large_straight",
            "full_house", "chance", "yatzy"
            ]
        
        self.action_to_index = {}
        self.counter = 0
        self.target_net = {}


    def decay_epsilon(self):

        """
        Decay the epsilon value using epsilon_decay parameter
        """

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    

    def choose_action(self, state):

        """
        Choose next action to take given the current state of the game

        Args:
            state: Environment state

        Returns:
            action: Returns the action chosen by the agent
        """
        
        X = self.state_to_input(state)
        A1,A2,A3 = self.forward_pass(X)

        self.net["A1"] = A1
        self.net["A2"] = A2
        self.net["A3"] = A3

        actions_index_list = self.available_action_indices(state)

        logits = A3.flatten()
        
        mask = np.full_like(logits, -np.inf)
        mask[actions_index_list] = logits[actions_index_list]

        if np.random.rand() < self.epsilon:
            action_index = np.random.choice(actions_index_list)
        else:
            action_index = np.argmax(mask) 

        action = self.actions[action_index]

        return action


    def update(self, state, action, reward, next_state):
        
        """
        Updates the Q network using backpropagation

        Args:
            state: Environment state
            action: Action chosen by the agent
            reward: Reward gotten from the action
            next_state: Environment state after the action

        """
        
        X = self.state_to_input(state)

        A1 = self.net["A1"] 
        A2 = self.net["A2"] 
        A3 = self.net["A3"]

        dW1, db1, dW2, db2, dW3, db3 = self.backward_pass(X, A1, A2, A3, action, reward, next_state)

        self.update_weights(dW1, db1, dW2, db2, dW3, db3)

    def update_batch(self, batch):

        states, actions, rewards, next_states = batch

        for i in range(len(states)):

            state = states[i]
            action = actions[i]
            reward = rewards[i]
            next_state = next_states[i]

            self.update(
                state,
                action,
                reward,
                next_state
            )


    def state_to_input(self, state):

        """
        Convert state of the game to a format that can be given to the neural network
        for forward pass

        Args:
            state: Environment state

        Returns:
            x: Returns the game state in a numpy list
        """
        
        
        dice, rolls_left, scorecard_mask = state

        dice = np.array(dice)
        scorecard_mask = np.array(scorecard_mask)
        dice_counts = np.bincount(dice, minlength=7)[1:] / 5
        roll_encoding = np.zeros(3)
        roll_encoding[rolls_left] = 1

        x = np.concatenate([dice_counts, roll_encoding, scorecard_mask])
        
        return x.reshape(-1, 1)
    

    def initalize_actions(self):

        """
        Initialize actions map that has mapping between index
        and an action 
        i.e. {1: ("reroll", np.array([0,0,0,0,0]), 2: ("reroll", np.array([1,0,0,0,0]) ... }
        """

        self.actions = {}

        i = 0
        
        for p in itertools.product([0, 1], repeat=5):
            self.actions[i] = ("reroll", np.array(p))
            i += 1

        for category in self.categories:
            self.actions[i] = ("score", category)
            i += 1

        self.action_to_index = {
            (a[0], tuple(a[1]) if a[0] == "reroll" else a[1]): i
            for i, a in self.actions.items()
        }


    def available_actions(self, state):
        
        """
        Returns a list of available actions to take given the game state.
        (i.e you cant take reroll actions if you have no rerolls left for the current turn.)

        Args:
            state: Environment state

        Returns:
            actions: returns list of all available actions given current state
        """

        dice, rolls_left, scorecard_mask = state

        actions = []

        if rolls_left > 0:
            for numbers in itertools.product([0, 1], repeat=5):
                actions.append(("reroll", np.array(numbers)))

        if rolls_left == 0:
            for i in range(len(scorecard_mask)):
                if scorecard_mask[i] == 0:
                    actions.append(("score", self.categories[i]))

        return actions
    

    def available_action_indices(self, state):

        """
        Returns a list of available actions to take given the game state.
        (i.e you cant take reroll actions if you have no rerolls left for the current turn.)

        Args:
            state: Environment state

        Returns:
            index_list: returns indicies of all available actions given the state
        """

        actions = self.available_actions(state)
        
        index_list = []

        for action in actions:
            action_name = (action[0], tuple(action[1]) if action[0] == "reroll" else action[1])
            index_list.append(self.action_to_index[action_name])
        
        return index_list


    def initialize(self):

        """
        Initialize weights for the neural network layers using HE-initialization (Kaiming initialization)
        Normal distribution with variance of 2/N_in where N_in is input size for that layer

        """
        
        self.net["W1"] = np.random.normal(0, np.sqrt(2.0 / 24), (128, 24))
        self.net["W2"] = np.random.normal(0, np.sqrt(2.0 / 128), (128, 128))
        self.net["W3"] = np.random.normal(0, np.sqrt(2.0 / 128), (47, 128))

        self.net["b1"] = np.zeros((128,1))
        self.net["b2"] = np.zeros((128,1))
        self.net["b3"] = np.zeros((47,1))

        self.initalize_actions()

        # Target net is a frozen copy of the current neural network to stabilize training
        # when calculating target values using the bellman equation.
        self.target_net = copy.deepcopy(self.net)


    def forward_pass(self, X):

        """
        Neural network forward pass where input flows through the network to give an output

        Args:
            X: Environment state in numpy array format

        Returns:
            A1, A2, A3: Activations of each neural layer
        """        

        A1 = self.relu(self.net["W1"] @ X + self.net["b1"])
        A2 = self.relu(self.net["W2"] @ A1 + self.net["b2"])
        A3 = self.net["W3"] @ A2 + self.net["b3"]

        return A1,A2,A3
    

    def forward_pass_target(self, X):

        """
        Neural network forward pass where input flows through the network to give an output
        This function uses the frozen target network instead of the main network.

        Args:
            X: Environment state in numpy array format

        Returns:
            A1, A2, A3: Activations of each neural layer
        """  

        A1 = self.relu(self.target_net["W1"] @ X + self.target_net["b1"])
        A2 = self.relu(self.target_net["W2"] @ A1 + self.target_net["b2"])
        A3 = self.target_net["W3"] @ A2 + self.target_net["b3"]


        return A1,A2,A3
    

    def loss_vector(self, A3, action_index, reward, next_state):

        """
        Calculates the loss vector for the output

        Args:
            A3: Activation values for the final layer of the network
            action_index: Index of the action that was chosen in the previous state
            reward: Reward gotten for the previous action
            next_state: State of the game after the action in previous state

        Returns:
            result: loss vector for the output layer (loss calculated only for the action taken)
            ie. [0,0,0,0,0,-2.4563,0,0,0,0 ... ]
        """  
        
        y = self.bellman(next_state, reward)
       
        result = np.zeros_like(A3)
        result[action_index] = A3[action_index] - y

        return result
        

    def bellman(self, next_state, reward):

        """
        Calculates the bellman target value

        Args:
            next_state: The next state of the environment after action 
            reward: reward gotten from the previous action

        Returns:
            target: Target value
        """  

        self.counter += 1

        if self.counter >= 100:
            self.target_net = copy.deepcopy(self.net)
            self.counter = 0

        X = self.state_to_input(next_state)
        
        _,_,A3 = self.forward_pass_target(X)

        actions_index_list = self.available_action_indices(next_state)

        logits = A3.flatten()
        
        mask = np.full_like(logits, -np.inf)
        mask[actions_index_list] = logits[actions_index_list]
         
        next_Q_max = np.max(mask)

        target = reward + self.gamma * next_Q_max

        if all(x == 1 for x in next_state[2]):
            target = reward

        return target


    def backward_pass(self, X, A1, A2, A3, action, reward, next_state):

        """
        Neural network backward pass

        Args:
            X: Game state input to neural network as numpy list
            A1,A2,A3: Activations calculated in forward pass for each neural layer
            action: Action taken by agent
            reward: Reward gotten for previous action
            next_state: The next state of the environment after action

        Returns:
            dW1, db1, dW2, db2, dW3, db3: Gradients for each layer and bias vectors
        """  


        action_label, dice = action
        if action_label == "reroll":
            action = (action_label, tuple(dice))
        action_index = self.action_to_index[action]

        dA3 = self.loss_vector(A3, action_index, reward, next_state)

        dW3 = dA3 @ A2.T
        db3 = dA3
        dA2 = self.net["W3"].T @ dA3

        dZ2 = dA2 * (A2 > 0)

        dW2 = dZ2 @ A1.T
        db2 = dZ2
        dA1 = self.net["W2"].T @ dA2

        dZ1 = dA1 * (A1 > 0)
        
        dW1 = dA1 @ X.T
        db1 = dZ1

        dW1 = np.clip(dW1, -1, 1)
        dW2 = np.clip(dW2, -1, 1)
        dW3 = np.clip(dW3, -1, 1)
        db1 = np.clip(db1, -1, 1)
        db2 = np.clip(db2, -1, 1)
        db3 = np.clip(db3, -1, 1)
        
        return dW1, db1, dW2, db2, dW3, db3


    def update_weights(self, dW1, db1, dW2, db2, dW3, db3):

        """
        Update network weights with gradient descent

        Args:
            dW1, db1, dW2, db2, dW3, db3: Gradients calculated in backward pass
        """  

        self.net["W1"] -= self.lr * dW1
        self.net["b1"] -= self.lr * db1

        self.net["W2"] -= self.lr * dW2
        self.net["b2"] -= self.lr * db2

        self.net["W3"] -= self.lr * dW3
        self.net["b3"] -= self.lr * db3


    def relu(self, x):

        """
        RElu activation function

        Args:
            x: activations in a network layer

        Returns:
            returns the activations for inputs greater than zero, otherwise sets them to zero for negative values.
        """  
        return np.maximum(0,x)

