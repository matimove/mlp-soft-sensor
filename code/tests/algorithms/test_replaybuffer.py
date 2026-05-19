import numpy as np
from algorithms.replaybuffer import Buffer

def test_pushing_and_length():
    buffer = Buffer(capacity=10)
    buffer.push(1, 2, 3, 4)
    assert len(buffer) == 1

def test_buffer_sampling():
    buffer = Buffer(capacity=10)
    for i in range(10):
        buffer.push(i, i+1, i+2, i+3)

    states, actions, rewards, next_states = buffer.sample(4)

    assert len(states) == 4
    assert len(actions) == 4
    assert len(rewards) == 4
    assert len(next_states) == 4

def test_buffer_overflow():
    buffer = Buffer(capacity=3)
    for i in range(5):
        buffer.push(i, i, i, i)

    assert len(buffer) == 3