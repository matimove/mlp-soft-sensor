import numpy as np
from game.scoreboard import Scoreboard

def test_initialization():
    sb = Scoreboard()
    sb.initialize()
    assert sb.scoreboard is not None
    assert all(v is None for v in sb.scoreboard.values())
    assert sb.bonus_achieved is False

def test_upper_sum_no_bonus():
    sb = Scoreboard()
    sb.initialize()

    sb.scoreboard["ones"] = 1
    sb.scoreboard["twos"] = 2
    sb.scoreboard["threes"] = 3
    sb.scoreboard["fours"] = 4
    sb.scoreboard["fives"] = 5
    sb.scoreboard["sixes"] = 6
    complete, total, bonus = sb.get_upper_sum()
    assert complete is True
    assert total==21
    assert bonus==0

def test_place_score():
    sb = Scoreboard()
    sb.initialize()

    class DiceTest:
        def display(self):
            return np.array([1, 1, 2, 3, 4])

    dice = DiceTest()
    score = sb.place_score(dice, "ones")
    assert score==2
    assert sb.scoreboard["ones"]==2