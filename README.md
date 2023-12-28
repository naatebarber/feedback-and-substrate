# Feedback and Substrate

In reading I Am A Strange Loop by Doug Hofstader I thought it'd be cool to try and create a substrate (environment with simple set of rules) in which epiphenomena could occur. 

## Overview:  

Using the `Sim` class and the abstract `Rule` class, create a continuois space where Sim agents can constantly percieve and interact.

I'm currently using tkinter to visualize stuff but you really don't have to do all that. Try to get funky with CLI output if you wanna.

Every Rule will should extend the following methods:
  - `mutator: Callable[Sim]` -> Write and return a state change function that can operate on a passed sim (and it's essence).
  - `essence -> Any` -> Create a blank essence for a Sim
  - `step` -> Call the `.become_affected()` method on all sims, invoking the state change function on every sim. _Note: I intend to make some `become_affected` call to Sim N enact a similar state change in Sim M. Not fleshed out yet but interactions between sims need to occur for any meaning to arise out of this_.

Running your Simulation:  

```python
# Spawn in your rule.
# My EatRule takes in an argument for population size - how many sims are spawned
rule = EatRule(10)

# Spawn in your environment
# EatEnvironment takes as arguments a dim (500) and a perception distance - eg how much agar space the Sim can see in every direction
environment = EatEnvironment(500, 15)

# Helper method on my environment for creating the agar space.
environment.random_normal()

# Bins my environment to my rule
rule.set_environment(environment=environment)

# Spawns in sims with essence and mutators
rule.initialize()

# Runs steps every 0.01 seconds.
rule.run(0.01)
```
  

## List of Current Rules/Environments:
  - **Eat**: An Agar-esque environment where each sim exists as a hungry microbe trying to find and capture stationary food
    - _Environment_ is a `dim * dim` sized grid of integers.
      - An integer value of 1 symbolizes food
      - An integer value of 0 symbolizes empty space
      - An integer value of -1 symbolizes the edge of the map
    - _Essence_ (Sim state) is a class containing:
      - `perception`: `visibility * visilibity` slice of the aforementioned grid, with the Sim at the center. This is what the Sim's state change function (it's mutator in the code) uses to change it's and it's environments state.
      - `x & y`: The coordinates of the Sim on the environment
      - `environment`: A reference to the shared environment, allowing the mutator to update the Sim's surroundings after it performs an action
    - _Mutator_ (State change function) in its first iteration is an algorithm that moves the sim towards the closest mote of food, and once `sim.x == mote.x && sim.y == mote.y`, consumes the food, removing it from the environment.

## Rule Creation

### First steps would be:  
 - Create a set of rules
 - Create an environment (world state) that can be updated via that set of rules
 - Create Sims, entities that can percieve and mutate the environment by the rule set
 - Create a graphical interface displaying environment state and Sim behavior

### Next steps would be:  
 - **Algorithmic Feedback**: Add communication between sims. Be careful of overflow when a loop is established. Best way might be to have communication cycles operate between steps. Communication could be presented in the form of Sim A forcing perceptual hallucination upon Sim B. 
 - **Reinforcement Derived Feedback**: Add reinforcement learning to sims instead of having them follow an algorithm. Would be interesting to see what symbols the Sims develop without specific instruction and how they communicate.