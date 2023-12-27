from abc import abstractmethod, ABC
from typing import Callable, Any, List
from ipycanvas import Canvas, hold_canvas
from febas.sim import Sim
import time


class Rule(ABC):
    def __init__(self, pop_size: int):
        self.canvas = Canvas()
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

    def run(self, delay: float):
        while True:
            self.step()
            time.sleep(delay)
