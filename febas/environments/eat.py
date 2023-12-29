import random


class EatEnvironment:
    def __init__(
        self,
        dim: int,
        visibility: int,
        food_density: float = 0.01,
        food_spawn_rate: float = 0,
    ):
        self.dim = dim
        self.visibility = visibility
        self.food_density = food_density
        self.food_spawn_rate = food_spawn_rate
        self.agar = []

    def random_normal(self):
        self.agar = [
            [
                (1 if random.random() > (1 - self.food_density) else 0)
                for i in range(self.dim)
            ]
            for j in range(self.dim)
        ]

    def spawn_food(self):
        if random.random() < self.food_spawn_rate:
            x, y = (random.randint(0, self.dim - 1), random.randint(0, self.dim - 1))
            self.agar[x][y] = 1

    def perspective(self, x: int, y: int):
        xs = max(0, x - self.visibility)
        xe = min(self.dim, x + self.visibility + 1)
        ys = max(0, y - self.visibility)
        ye = min(self.dim, y + self.visibility + 1)

        self.spawn_food()

        perspective = []
        for r in range(x - self.visibility, x + self.visibility + 1):
            subrow = []
            for v in range(y - self.visibility, y + self.visibility + 1):
                if r < 0 or r > self.dim - 1:
                    subrow.append(-1)
                    continue
                if v < 0 or v > self.dim - 1:
                    subrow.append(-1)
                    continue

                subrow.append(self.agar[r][v])
            perspective.append(subrow)

        return perspective
