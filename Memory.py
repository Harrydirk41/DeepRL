import random
from collections import namedtuple



class Memory(object):
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.memory = []
        self.index = 0
        self.batch_size = batch_size
        self.cur_memory = namedtuple("cur_memory", ("states", "action", "rewards", "time_step"))

    def push(self, states, actions, rewards, time_step):
        for s,a,r,t in zip(states,actions,rewards,time_step):
            if (len(self.memory) < self.capacity):
                self.memory.append(None)
            self.memory[self.index] = self.cur_memory(s,a,r,t)
            self.index = (self.index + 1) % self.capacity

    def sampe(self):
        batch_sample = random.sample(self.memory, self.batch_size)
        batch = self.cur_memory(*zip(*batch_sample))
        return batch
