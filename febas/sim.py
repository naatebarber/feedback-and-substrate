from typing import Callable

# Extensible class that can affect itself and others


class Sim:
    def __init__(self, identity: int):
        self.identity = identity
        self.affects = []
        self.essence = None
        self.mutator = None

    def set_essence(self, essence):
        self.essence = essence

    def set_mutator(self, mutator: Callable):
        self.mutator = mutator

    def add_affects(self, sim):
        self.affects.append(sim)

    def become_affected(self):
        self.mutator(self)
        for sim in self.affects:
            sim.become_affected()
