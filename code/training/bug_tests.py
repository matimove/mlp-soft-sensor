import numpy as np
import pytest

from game.scoreboard import Scoreboard


@pytest.fixture
def scoreboard():
    return Scoreboard()


# ---------- Upper section ----------


def test_ones(scoreboard):
    dice = np.array([1, 1, 3, 4, 6])
    assert scoreboard.score_dice(dice, 'ones') == 2



def test_twos(scoreboard):
    dice = np.array([2, 2, 2, 5, 6])
    assert scoreboard.score_dice(dice, 'twos') == 6



def test_threes(scoreboard):
    dice = np.array([3, 3, 1, 5, 6])
    assert scoreboard.score_dice(dice, 'threes') == 6



def test_fours(scoreboard):
    dice = np.array([4, 4, 4, 1, 2])
    assert scoreboard.score_dice(dice, 'fours') == 12



def test_fives(scoreboard):
    dice = np.array([5, 5, 2, 3, 4])
    assert scoreboard.score_dice(dice, 'fives') == 10



def test_sixes(scoreboard):
    dice = np.array([6, 6, 6, 6, 1])
    assert scoreboard.score_dice(dice, 'sixes') == 24


# ---------- Pair tests ----------


def test_one_pair(scoreboard):
    dice = np.array([2, 2, 5, 5, 6])
    assert scoreboard.score_dice(dice, 'one_pair') == 10



def test_one_pair_none(scoreboard):
    dice = np.array([1, 2, 3, 4, 5])
    assert scoreboard.score_dice(dice, 'one_pair') == 0



def test_two_pair(scoreboard):
    dice = np.array([2, 2, 5, 5, 6])
    assert scoreboard.score_dice(dice, 'two_pair') == 14



def test_two_pair_none(scoreboard):
    dice = np.array([1, 1, 1, 4, 5])
    assert scoreboard.score_dice(dice, 'two_pair') == 0

    assert scoreboard.score_dice(dice, 'invalid_pick') is None