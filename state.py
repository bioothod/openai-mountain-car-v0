import numpy as np

from collections import deque

class state(object):
    def __init__(self, shape, size):
        self.steps = deque()
        self.shape = shape
        self.size = size
        self.value = None

        for i in range(size):
            self.push_zeroes()

    def push_zeroes(self):
        self.push_array(np.zeros(self.shape))

    def push_array(self, step_array):
        assert self.shape == step_array.shape[0]

        if len(self.steps) == self.size:
            self.steps.popleft()

        self.steps.append(step_array)

    def complete(self):
        self.value = np.concatenate(self.steps)

    def read(self):
        return self.value

    def reshape(self, rows, cols):
        return self.value.reshape(rows, cols)

    def vector(self):
        return self.value.reshape(1, self.value.shape[0])

