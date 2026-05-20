import numpy as np
from game.yatzy import Yatzy


def test_resetting_game():
    game = Yatzy()
    state = game.reset()

    assert len(state[0]) == 5 #check dice
    assert state[1] == 2 #check rerolls
    assert len(state[2]) == 15 #check scoreboard
    assert game.done is False


def test_rolls_left_goes_down():
    game = Yatzy()
    game.reset()

    action = ("reroll", np.array([0, 0, 0, 0, 0]))
    game.step(action)

    assert game.rolls_left == 1


def test_scoring_action():
    game = Yatzy()
    game.reset()

    action = ("score", "ones")
    next_state, reward, done, final_score, info = game.step(action)

    assert game.scoreboard.scoreboard["ones"] is not None
    assert np.isscalar(reward)


def test_game_over_scoreboard_full():
    game = Yatzy()
    game.reset()

    for category in game.scoreboard.scoreboard.keys():
        game.scoreboard.scoreboard[category] = 1

    game.check_if_game_over()

    assert game.done is True