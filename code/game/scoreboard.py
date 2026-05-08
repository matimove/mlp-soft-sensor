import numpy as np

class Scoreboard:

    """
    Class that implements the scoreboard functionality for the yatzy game

    Args:
        scoreboard: Holds scores with {category: score achieved} mapping
    """  

    def __init__(self):
        self.scoreboard = None
    
    def initialize(self):

        """
        Initializes the scoreboard at the start of each game

        Args:
            scoreboard: Holds scores with {category: score achieved} mapping
        """  
         
        self.scoreboard = {"ones": None,
                           "twos": None,
                           "threes": None,
                           "fours": None,
                           "fives": None,
                           "sixes": None,
                           "one_pair": None,
                           "two_pair": None,
                           "three_of_a_kind": None,
                           "four_of_a_kind": None,
                           "small_straight": None,
                           "large_straight": None,
                           "full_house": None,
                           "chance": None,
                           "yatzy": None,
                           }

        #self.categories = ["ones", "twos", "threes", "fours", "fives", "sixes",
        #    "one_pair", "two_pairs", "three_of_a_kind", "four_of_a_kind",
        #    "small_straight", "large_straight",
        #    "full_house", "chance", "yatzy"]
    
    def return_scoreboard(self):
        """
        Function to return current scoreboard
        """  
        return self.scoreboard
    
    def get_upper_sum(self):

        """
        Calculates sum of the upper section scores (In yatzy you get 50 points bonus if the upper
        combos sum up to 63 or more.)

        """  

        upper_section = [self.scoreboard["ones"],
                    self.scoreboard["twos"],
                    self.scoreboard["threes"],
                    self.scoreboard["fours"],
                    self.scoreboard["fives"],
                    self.scoreboard["sixes"]
                    ]
    
        upper_sum = sum(score if score is not None else 0 for score in upper_section)
        
        complete = True

        for item in upper_section:
            if item is None:
                complete = False

        bonus_score = 0

        if complete and upper_sum >= 63 and not self.bonus_achieved:
            print("BONUS ACHIEVED!")
            self.bonus_achieved = True
            bonus_score += 100

        return (complete, upper_sum, bonus_score)
    
    def score_game(self):

        """
        Calculates the final score when scoreboard is full.

        """  
    
        top_section_score = sum([
            self.scoreboard["ones"],
            self.scoreboard["twos"], 
            self.scoreboard["threes"], 
            self.scoreboard["fours"], 
            self.scoreboard["fives"], 
            self.scoreboard["sixes"]
            ])
        
        if top_section_score >= 63:
            bonus = 50
        else:
            bonus = 0
        
        total_score = (sum(list(self.scoreboard.values())) + bonus, bonus == 50)

        return total_score
    
    def get_scoreboard_mask(self):
        """
        Checks which combinations have already been used and which ones are still free

        Return:
            list of values 0 and 1 depending on if the scoreboard place is free or used
        """  
        return [0 if item is None else 1 for item in self.scoreboard.values()]
    
    def place_score(self, dice, category_name):

        """
        Places a score to the category

        Args:
            dice: Current dice
            category_name: category name that you want to place the score into

        Returns:
            Score achieved for the category
        """  
        
        self.scoreboard[category_name] = self.score_dice(dice.display(), category_name)
        return self.score_dice(dice.display(), category_name)
           
    def count_dice(self, dice):

        """
        Counts how many of each dice value there are (i.e {1:0, 2:2, 3:1 ...})

        Args:
            dice: current dice

        Returns:
            dct: dictionary with counts of each dice value
            (i.e dice: [1,1,1,2,4] --> {1:3, 2:1, 3:0 4:1, 5:0, 6:0})
        """  

        dct = {}
        for d in dice:
            if d in dct:
                dct[d] += 1
            else:
                dct[d] = 1
        return dct
    
    def score_dice(self, dice, pick):

        """
        Calculates score for the chosen category given the current dice

        Args:
            dice: current dice
            pick: which placement in scoreboard you want to put the dice into

        Returns:
            score achieved in chosen category with the given dice
        """  
        
        if pick == 'ones':
            return np.sum(dice == 1) * 1
        
        if pick =='twos':
            return np.sum(dice == 2)*2
        
        if pick =='threes':
            return np.sum(dice == 3)*3
        
        if pick =='fours':
            return np.sum(dice == 4)*4
        
        if pick =='fives':
            return np.sum(dice == 5)*5
        
        if pick =='sixes':
            return np.sum(dice == 6)*6
        
        if pick =='one_pair':
            counts = np.bincount(dice, minlength=7)

            pairs = np.where(counts >= 2)[0]  

            if len(pairs) == 0:
                return 0

            highest = pairs.max()
            return 2 * highest
        
        if pick =='two_pair':
            counts = np.bincount(dice, minlength=7)

            pairs = np.where(counts >= 2)[0]

            if pairs.size < 2:
                return 0

            top_two = np.sort(pairs)[-2:]
            return 2 * top_two.sum()
        
        if pick =='three_of_a_kind':
            counts = np.bincount(dice, minlength=7)

            triples = np.where(counts >= 3)[0]

            if triples.size == 0:
                return 0

            return 3 * triples.max()
        
        if pick =='four_of_a_kind':
            counts = np.bincount(dice, minlength=7)

            quads = np.where(counts >= 4)[0]
            
            if quads.size == 0:
                return 0

            return 4 * quads.max()
        
        if pick =='small_straight':
            return 15 if np.array_equal(dice, np.array([1,2,3,4,5])) else 0
        
        if pick =='large_straight':
            return 20 if np.array_equal(dice, np.array([2,3,4,5,6])) else 0
        
        if pick =='full_house':
            counts = np.bincount(dice, minlength=7)
            values = np.where(counts > 0)[0]
            freqs = counts[values]

            if len(values) == 2 and sorted(freqs) == [2, 3]:
                return np.sum(dice)

            return 0
                
        if pick =='chance':
            return np.sum(dice)
        
        if pick =='yatzy':
            return 50 if np.all(dice == dice[0]) else 0