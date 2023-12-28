from febas.rules.eat_rule import EatEnvironment, EatRule
from febas.rules.reinforcement_eat_rule import ReinforcementEatRule


def algorithmic_eat():
    rule = EatRule(10)
    environment = EatEnvironment(500, 15, food_density=0.01)
    environment.random_normal()

    rule.set_environment(environment=environment)

    rule.initialize()

    rule.run(0.01)


def reinforcement_eat():
    rule = ReinforcementEatRule(1)
    environment = EatEnvironment(500, 16, food_density=0.02)
    environment.random_normal()

    rule.set_environment(environment=environment)
    rule.initialize()
    rule.run(0)


if __name__ == "__main__":
    # algorithmic_eat()
    reinforcement_eat()
