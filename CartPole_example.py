import gym
import time
from GANN import GANN

env = gym.make('CartPole-v1')
a = GANN([4, 10, 10, 1])  #architecture of NN

POPULATION = 5
GENERATIONS = 50
a.generatePopulation(POPULATION)

#evolution phase
for generation in range(GENERATIONS):
    a.reset()   #reset rewards
    print("generation: ", generation)

    for i in range(POPULATION):
        obs = env.reset()
        for _ in range(400):    #400 time steps
            action = a.forwardPropagate(obs, i)     #figure out action by forward propagating inputs through NN

            if action > 0.5:
                action = 1
            else:
                action = 0

            obs, reward, done, info = env.step(action)  #perform action
            a.updateRewards(reward, i)      #update rewards for ith weight in population
            # if (i == 0 or i == 5 or i == 9):
            #     env.render()
            #     time.sleep(0.02)

    # print(a.rewards)
    a.evolve(0.5, 1)

# render best of the best
obs = env.reset()
for _ in range(400):
    action = a.forwardPropagate(obs)
    if action > 0.5:
        action = 1
    else:
        action = 0

    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.02)

env.close()
a.evolution()


