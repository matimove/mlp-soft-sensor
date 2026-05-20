import numpy as np
from algorithms.neural_net import NN

def test_forward_pass_output():
    nn = NN()
    nn.initialize()
    state = ((1, 1, 3, 4, 5), 2, [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

    X = nn.state_to_input(state)
    A1, A2, A3 = nn.forward_pass(X)

    assert np.all(np.isfinite(A3))

def test_weights_change():
    np.random.seed(0)
    nn = NN()
    nn.initialize()

    state = ((1, 1, 3, 4, 5), 2, [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    next_state = ((1, 1, 1, 3, 5), 1, [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

    action = nn.choose_action(state)
    old_weights = nn.net["W1"].copy()
    nn.update(state, action, 2, next_state)

    assert not np.array_equal(old_weights, nn.net["W1"])

def test_many_updates():
    np.random.seed(0)
    nn = NN()
    nn.initialize()

    state = ((1, 1, 3, 4, 5), 2, [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    next_state = ((1, 1, 1, 3, 5), 1, [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

    for _ in range(10):
        action = nn.choose_action(state)
        nn.update(state, action, 2, next_state)

    assert True


def test_forward_pass_same_inputs():
    np.random.seed(0)
    nn = NN()
    nn.initialize()

    state = ((1, 1, 3, 4, 5), 2, [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

    X = nn.state_to_input(state)
    A1_1, A2_1, A3_1 = nn.forward_pass(X)
    A1_2, A2_2, A3_2 = nn.forward_pass(X)

    assert np.allclose(A3_1, A3_2)

def test_q_value_increases():
    np.random.seed(0)

    nn = NN()
    nn.initialize()

    state = ((1, 1, 1, 1, 1), 0, [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    next_state = ((1, 1, 1, 3, 5), 2, [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])

    action = ("score", "yatzy")

    X = nn.state_to_input(state)

    A1, A2, A3_before = nn.forward_pass(X)

    action_index = nn.action_to_index[action]

    before_Q = A3_before[action_index][0]

    for _ in range(100):
        nn.choose_action(state)
        nn.update(state, action, 2, next_state)

    A1, A2, A3_after = nn.forward_pass(X)

    after_Q = A3_after[action_index][0]

    assert after_Q > before_Q

def test_bellman_game_end_reward():
    nn = NN()
    nn.initialize()

    game_end_state = ((1,1,1,1,1),2,[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    
    target = nn.bellman(game_end_state, 50)

    assert target == 50