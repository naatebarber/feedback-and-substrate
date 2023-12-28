import random
from typing import Any, Callable
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy
import math

from febas.rules.eat_rule import EatEnvironment, EatRule, Essence
from febas.sim import Sim


class Brain(nn.Module):
    def __init__(self, visibility: int):
        super(Brain, self).__init__()
        d_in = math.pow(visibility * 2 + 1, 2)
        d_in = int(d_in)
        # choice (x, -x, y, -y)
        d_out = 4

        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(10 * 15 * 15, 50)
        self.fc2 = nn.Linear(50, d_out)

        self.optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        self.optimizer = optim.Adam(params=self.parameters(), lr=0.0001)

        self.last_call = None

    def forward(self, x):
        x = torch.FloatTensor(x).unsqueeze(0).unsqueeze(0)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 10 * 15 * 15)
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
        return q_values

    def optim_step(self, state, action, reward, next_state, gamma):
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        reward = torch.tensor([reward], dtype=torch.float32)
        action = torch.tensor([action], dtype=torch.int64)

        q_values = self.forward(state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        next_q_values = self.forward(next_state).detach()
        max_next_q_value = next_q_values.max(1)[0]
        target_q_value = reward + gamma * max_next_q_value

        loss = F.mse_loss(q_value, target_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class NeuralEssence(Essence):
    def __init__(self, env: EatEnvironment):
        super().__init__(env)

    def with_brain(self, b: Brain):
        self.brain = b
        self.step_counter = 0


class ReinforcementEatRule(EatRule):
    def mutator(self) -> Callable[[Sim], Any]:
        def hungry_sim(sim: Sim):
            ess: NeuralEssence = sim.essence
            ess.get_perspective()
            first_perspective = deepcopy(ess.perspective)

            decision = ess.brain.forward(torch.FloatTensor(first_perspective))

            # Epsilon greedy strat
            epsilon = 0.2
            choice = 0
            if random.random() < 0.2:
                choice = np.random.choice(range(4))
            else:
                choice = torch.argmax(decision).item()

            if choice == 0 and ess.x < ess.env.dim - 1:
                ess.x += 1
            elif choice == 1 and ess.x > 0:
                ess.x -= 1
            elif choice == 2 and ess.y < ess.env.dim - 1:
                ess.y += 1
            elif choice == 3 and ess.y > 0:
                ess.y -= 1

            reward = 0
            if self.environment.agar[ess.x][ess.y] == 1:
                print("Sim ATE")
                reward = 5
                self.environment.agar[ess.x][ess.y] = 0

            ess.get_perspective()
            second_perspective = deepcopy(ess.perspective)
            ess.brain.optim_step(
                first_perspective, choice, reward, second_perspective, gamma=0.2
            )

            ess.step_counter += 1

            if ess.step_counter % 1000 == 0:
                print("Checkpoint")
                torch.save(ess.brain.state_dict(), "./q-sim.pth")

        return hungry_sim

    def essence(self) -> Any:
        essence = NeuralEssence(self.environment)
        brain = Brain(visibility=self.environment.visibility)

        try:
            sd = torch.load("./q-sim.pth")
            print("Resuming q sim")
            brain.load_state_dict(sd)
        except:
            print("Starting fresh")

        essence.with_brain(brain)

        return essence
