from febas.rules.eat_rule import EatEnvironment, EatRule
from febas.rules.reinforcement_eat_rule import ReinforcementEatRule


def algorithmic_eat():
    rule = EatRule(10)
    environment = EatEnvironment(500, 15, food_density=0.01)
    environment.random_normal()

    rule.set_environment(environment=environment)

    rule.run(0.01)


def reinforcement_eat():
    rule = ReinforcementEatRule(1, learn=True)
    environment = EatEnvironment(500, 16, food_density=0.04, food_spawn_rate=0.03)
    environment.random_normal()

    rule.set_environment(environment=environment)
    rule.run(delay=0, render=True)


if __name__ == "__main__":
    # algorithmic_eat()
    reinforcement_eat()
