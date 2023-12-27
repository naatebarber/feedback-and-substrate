from febas.rules.eat_rule import EatEnvironment, EatRule

if __name__ == "__main__":
    rule = EatRule(10)
    environment = EatEnvironment(500, 15)
    environment.random_normal()

    rule.set_environment(environment=environment)

    rule.initialize()

    rule.run(0.01)
