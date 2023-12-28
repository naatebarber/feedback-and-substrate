from abc import abstractmethod, ABC
from typing import Callable, Any, List
from febas.sim import Sim
import time


class Rule(ABC):
    def __init__(self, pop_size: int):
        self.pop_size = pop_size
        self.sims: List[Sim] = []
        self.environment = None

    def set_environment(self, environment: Any):
        self.environment = environment

    @abstractmethod
    def mutator(self) -> Callable[[Sim], Any]:
        pass

    @abstractmethod
    def essence(self) -> Any:
        pass

    @abstractmethod
    def step():
        """
        Update the sims and render with canvas
        """
        pass

    def initialize(self):
        for i in range(self.pop_size):
            s = Sim(identity=i)
            s.set_essence(self.essence())
            s.set_mutator(self.mutator())
            self.sims.append(s)

    @abstractmethod
    def initialize_render(self):
        pass

    def run(self, delay: float, render=True):
        self.initialize()
        if render:
            self.initialize_render()
        while True:
            self.step(render=render)
            time.sleep(delay)
