import random
from typing import Any, Callable
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy
import math

from febas.environments.eat import EatEnvironment
from febas.rules.eat_rule import EatRule, Essence
from febas.sim import Sim

import sys


class Brain(nn.Module):
    def __init__(self, visibility: int):
        super(Brain, self).__init__()
        if torch.backends.mps.is_available():
            print("using mps")
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Turn visibility into the xy dimension of the input space
        d_in = visibility * 2 + 1
        # choice (x, -x, y, -y)
        d_out = 4

        k_size = 3
        feature_maps = 30
        feature_maps_2 = 60

        # Input size:
        # [1, 1, visibility + 1, visibility + 1]
        self.conv1 = nn.Conv2d(1, feature_maps, kernel_size=k_size).to(self.device)
        self.conv2 = nn.Conv2d(feature_maps, feature_maps_2, kernel_size=k_size).to(
            self.device
        )

        reduced_size = d_in - ((k_size - 1) * 2)
        conv_output_size = feature_maps_2 * (reduced_size**2)

        self.fc1 = nn.Linear(conv_output_size, 12).to(self.device)
        self.fc2 = nn.Linear(12, 96).to(self.device)
        self.fc3 = nn.Linear(96, d_out).to(self.device)

        self.optimizer = optim.SGD(self.parameters(), lr=0.005, momentum=0.9)

    def forward(self, x):
        x = (
            torch.tensor(x, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

    def optim_step(self, state, action, reward, next_state, gamma):
        reward = F.relu(torch.tensor([reward], dtype=torch.float32, device=self.device))
        action = torch.tensor([action], dtype=torch.int64, device=self.device)

        q_values = self.forward(state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        print(q_value)

        next_q_values = self.forward(next_state).detach()
        # print(next_q_values)
        max_next_q_value = next_q_values.max(1)[0]
        # print(max_next_q_value)
        target_q_value = reward + gamma * max_next_q_value
        # print(target_q_value)

        loss = F.mse_loss(q_value, target_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class NeuralEssence(Essence):
    def __init__(self, env: EatEnvironment, learn=True):
        super().__init__(env)
        self.learn = learn
        self.eat_count = 0

    def with_brain(self, b: Brain):
        self.brain = b
        self.step_counter = 0


class ReinforcementEatRule(EatRule):
    def __init__(self, pop_size: int, learn=True):
        super().__init__(pop_size)
        self.learn = learn

    def mutator(self) -> Callable[[Sim], Any]:
        def hungry_sim(sim: Sim):
            ess: NeuralEssence = sim.essence
            ess.get_perspective()
            first_perspective = deepcopy(ess.perspective)

            decision = ess.brain.forward(first_perspective)

            # Vars
            mote_ate_reward = 1
            epsilon_decay = 0.999995
            min_epsilon = 0.12

            # Epsilon greedy strat
            choice = 0
            if random.random() < ess.epsilon:
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

            if ess.epsilon > min_epsilon:
                ess.epsilon *= epsilon_decay

            # Add reward for eating
            reward = 0
            if self.environment.agar[ess.x][ess.y] == 1:
                print("Sim ATE")
                ess.eat_count += mote_ate_reward
                reward += mote_ate_reward
                self.environment.agar[ess.x][ess.y] = 0

            ess.get_perspective()
            second_perspective = deepcopy(ess.perspective)
            if ess.learn:
                ess.brain.optim_step(
                    first_perspective, choice, reward, second_perspective, gamma=0.6
                )

            ess.step_counter += 1

            if ess.step_counter % 1000 == 0:
                print("Checkpoint")
                print("Total ate:", ess.eat_count)
                print("Current reward:", reward)
                print("Current epsilon:", ess.epsilon)
                torch.save(ess.brain.state_dict(), "./q-sim.pth")

        return hungry_sim

    def essence(self) -> Any:
        essence = NeuralEssence(self.environment, learn=self.learn)
        essence.epsilon = 0.2

        brain = Brain(visibility=self.environment.visibility)

        try:
            model_pth = sys.argv[1]
            print(model_pth)
            sd = torch.load(model_pth)
            print("Resuming q sim")
            brain.load_state_dict(sd)
            # Set epsilon to lowest, assuming it bottomed out during training
            essence.epsilon = 0.05
        except:
            print("Starting fresh")

        essence.with_brain(brain)

        return essence
