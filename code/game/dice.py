import random
import numpy as np

class Dice:

    """
        Class that implements dice functionality for yatzy

        Args:
            dice: numpy array of five numbers (dice)
            lock_list: numpy array of five numbers (either (0 = reroll) or (1 = keep))
    """  
    
    def __init__(self):
        self.dice = None
        self.lock_list = None

    def reroll(self):

        """
        Rerolls dice that aren't locked in the lock list
        """  

        for i in range(5):
            if self.lock_list[i] == 0:
                self.dice[i] = random.randint(1,6)
        self.dice = np.sort(self.dice)

    def initialize(self):
        """
        Initalizes dice and lock_list between rounds
        """  
        self.dice = np.random.randint(1, 7, size=5)
        self.dice = np.sort(self.dice)
        self.lock_list = np.zeros(5)

    def display(self):
        """
        Function to return the current dice
        """  
        return self.dice
    
    def lock(self, lock_list):
        """
        Function to set the given list as the current lock_list
        
        Args:
        lock_list: list of five numbers 0 or 1 (i.e [0,1,0,0,1])

        """  
        self.lock_list = lock_list