import random
import math
import tkinter as tk
from typing import Callable, Any
import numpy as np

from febas.rules.base import Rule
from febas.sim import Sim
from febas.environments.eat import EatEnvironment


class Essence:
    def __init__(self, env: EatEnvironment):
        self.env = env
        self.x = random.randrange(0, env.dim)
        self.y = random.randrange(0, env.dim)
        self.perspective = env.perspective(self.x, self.y)

    def get_perspective(self):
        self.perspective = self.env.perspective(self.x, self.y)


class EatRule(Rule):
    def mutator(self) -> Callable[[Sim], Any]:
        def hungry_sim(s: Sim):
            ess: Essence = s.essence

            ess.get_perspective()

            random_direction = lambda: (
                random.randrange(-1, 2),
                random.randrange(-1, 2),
            )

            perspective = ess.perspective

            food = [
                (x, y)
                for x, i in enumerate(perspective)
                for y, v in enumerate(i)
                if v == 1
            ]
            sim_pos = math.floor(len(perspective) / 2)
            # The way I shift here confuses the SIMS when perspective gets truncated near the edge
            food_distance = [(x - sim_pos, y - sim_pos) for x, y in food]
            food_distance.sort(
                key=lambda x: math.sqrt(math.pow(x[0], 2) + math.pow(x[1], 2))
            )

            closest_food = food_distance[0] if len(food_distance) > 0 else None

            xd = None
            yd = None

            if not closest_food or random.random() < 0.1:
                xd, yd = random_direction()
            else:
                sim_pos = math.floor(len(perspective) / 2) + 1
                xd = closest_food[0]
                yd = closest_food[1]

            if ess.env.agar[ess.x][ess.y] == 1:
                print("Sim ate")
                ess.env.agar[ess.x][ess.y] = 0

            if xd > 0 and ess.x < ess.env.dim - 1:
                ess.x += 1
            elif xd < 0 and ess.x > 1:
                ess.x -= 1

            if yd > 0 and ess.y < ess.env.dim - 1:
                ess.y += 1
            elif yd < 0 and ess.y > 1:
                ess.y -= 1

        return hungry_sim

    def essence(self) -> Any:
        return Essence(self.environment)

    def initialize(self):
        super().initialize()

    def initialize_render(self):
        self.window = tk.Tk()
        self.canvas = tk.Canvas(
            self.window,
            {
                "width": self.environment.dim,
                "height": self.environment.dim,
                "bg": "#000000",
            },
        )
        self.canvas.pack()
        self.window.update()

    def step(self, render=True):
        if render:
            self.canvas.delete("all")

        if render:
            for i, x in enumerate(self.environment.agar):
                for j, y in enumerate(x):
                    if y == 1:
                        self.canvas.create_rectangle(
                            i, j, i + 1, j + 1, outline="white"
                        )

        for sim in self.sims:
            sim.become_affected()
            if render:
                self.canvas.create_rectangle(
                    sim.essence.x - self.environment.visibility,
                    sim.essence.y - self.environment.visibility,
                    sim.essence.x + self.environment.visibility,
                    sim.essence.y + self.environment.visibility,
                    outline="blue",
                )

        if render:
            self.window.update()
