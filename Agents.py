import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
from Iterations import ValueIteration
from Iterations import PolicyIteration
import matplotlib.pyplot as plt


class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


class VIagent(object):
    def __init__(self, action_space, epsilon, gamma, Nplan):
        self.action_space = action_space
        self.pi = np.array(ValueIteration(Nplan, gamma, epsilon))

    def act(self, observation, reward, done):
        state = env.getStateFromObs(obs)
        return (self.pi[state])


class PIagent(object):
    def __init__(self, action_space, epsilon, gamma, Nplan):
        self.action_space = action_space
        states, P = env.getMDP()
        self.pi = np.zeros(len(states))
        self.pi = np.array(PolicyIteration(Nplan, gamma, epsilon))

    def act(self, observation, reward, done):
        state = env.getStateFromObs(obs)
        return (self.pi[state])


Plans = [0,1,2,3,4,5,6,7,8,10]
Lavgsum, Lavgj = np.zeros((3,11)), np.zeros((3,11))
for NPlan in Plans:
    if __name__ == '__main__':
        env = gym.make("gridworld-v0")
        env.setPlan("gridworldPlans/plan"+str(NPlan)+".txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
        env.seed(0)  # Initialise le seed du pseudo-random
        statedic, mdp = env.getMDP()  # recupere le mdp : statedic
        state, transitions = list(mdp.items())[0]

        # Execution avec un Agent
        for a in range(3):
            if a == 0:
                agent = RandomAgent(env.action_space)
            if a == 1:
                agent = VIagent(env.action_space, 10e-7, 0.99, NPlan)
            if a == 2:
                agent = PIagent(env.action_space, 10e-7, 0.99, NPlan)

            episode_count = 10
            reward = 0
            done = False
            rsum = 0
            FPS = 0.000001
            totsum = 0
            totj = 0
            for i in range(episode_count):
                obs = env.reset()
                #env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
                #if env.verbose:
                #env.render(FPS, mode='human')
                j = 0
                rsum = 0
                while True:
                    action = agent.act(obs, reward, done)
                    obs, reward, done, _ = env.step(action)
                    rsum += reward
                    j += 1
                    #if env.verbose:
                    #env.render(FPS, mode='human')
                    if done:
                        totsum += rsum
                        totj += j
                        print(totsum)
                        print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                        break

            avg_rsum = totsum/episode_count
            avg_j = totj/episode_count
            Lavgsum[a, NPlan] = avg_rsum
            Lavgj[a, NPlan] = avg_j
            a += 1
            env.close()

print(Lavgsum)
print(Lavgj)

LavgsumMax = [1, 2, 2, 1, 0, 2, 2, 3, 1, 1]
#LavgjMax = [4, 13, 5, 24, 42, 42, 38, ]
"""
for NPlan in Plans:
    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan" + str(NPlan) + ".txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    env.render(mode='human')
"""


x = np.array(Plans)

y = LavgsumMax
z = Lavgsum[1, :].tolist()
del(z[9])
k = Lavgsum[0, :].tolist()
del(k[9])

ax = plt.subplot(111)
ax.bar(x-0.2, y, width=0.2, color='b', align='center', label='Max Rsum')
ax.bar(x, z, width=0.2, color='g', align='center', label='Avg VI Rsum')
ax.bar(x+0.2, k, width=0.2, color='r', align='center', label='Avg Random Rsum')
ax.legend(loc = 'best')
plt.show()