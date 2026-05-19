import random
from collections import deque
import numpy as np

class Buffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):

        self.buffer.append((
            state,
            action,
            reward,
            next_state
        ))

    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states = zip(*batch)

        return (
            states,
            actions,
            rewards,
            next_states,
        )

    def __len__(self):
        return len(self.buffer)