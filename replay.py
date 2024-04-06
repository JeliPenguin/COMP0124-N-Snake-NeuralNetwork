from collections import deque, namedtuple
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory():

    def __init__(self, capacity:int,batch_size:int):
        self.memory = deque([], maxlen=capacity)
        self.batch_size = batch_size

    def ready(self):
        return len(self.memory) >= self.batch_size

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def clear(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)