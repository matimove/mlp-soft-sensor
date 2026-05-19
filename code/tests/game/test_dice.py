import numpy as np
from game.dice import Dice

def test_initialization():
    dice = Dice()
    dice.initialize()
    assert dice.dice is not None
    assert len(dice.dice) == 5
    assert np.all((dice.dice >= 1) & (dice.dice <= 6))
    assert np.all(dice.lock_list == 0)

def test_locking_dice():
    dice = Dice()
    dice.initialize()
    lock = np.array([1, 1, 0, 1, 0])
    dice.lock(lock)
    assert np.array_equal(dice.lock_list, lock)

def test_lock_works_for_rerolling():
    np.random.seed(0)
    dice = Dice()
    dice.initialize()
    original = dice.dice.copy()
    lock = np.array([1, 1, 0, 0, 0])
    dice.lock(lock)
    dice.reroll()

    for i in range(5):
        if lock[i] == 1:
            assert dice.dice[i] == original[i]

def test_display_dice():
    dice = Dice()
    dice.initialize()
    assert np.array_equal(dice.display(), dice.dice)