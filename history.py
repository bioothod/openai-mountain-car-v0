from collections import deque
import numpy as np

class history_object(object):
    def __init__(self, o, w):
        self.o = o
        self.w = w

class history(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.history = deque()

        self.p = deque()
        self.p_sum = 0.0

    def clear(self):
        self.history = deque()
        self.p_sum = 0.0
        self.p = deque()

    def last(self, n):
        if n <= 0:
            return deque()

        start = 0
        if len(self.history) >= n:
            start = len(self.history) - n

        ret = deque()
        for i in range(start, len(self.history)):
            ret.append(self.history[i].o)
        
        return ret

    def size(self):
        return len(self.history)

    def full(self):
        return self.size() >= self.max_size

    def append(self, e, w):
        qlen = len(self.history) + 1
        if qlen > self.max_size:
            for i in range(qlen - self.max_size):
                self.p_sum -= self.history[0].w
                self.p.popleft()
                self.history.popleft()

        self.history.append(history_object(e, w))
        self.p.append(w)
        self.p_sum += w

    def sort(self):
        self.history = deque(sorted(self.history, key=lambda x: x.w))

    def get(self, idx):
        return self.history[idx].o

    def sample(self, size):
        idx = range(self.size())

        p = np.array(self.p) / self.p_sum
        ch = np.random.choice(idx, min(size, self.size()), p=p)

        ret = deque()
        for i in ch:
            ret.append(self.history[i].o)
        
        return ret
