import matplotlib

matplotlib.use("TkAgg")
import gym
import ast
import gridworld
from gym import wrappers, logger
import numpy as np
import copy

def ValueIteration(NPlan):

    # Creation de l'environnement
    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan"+str(NPlan)+".txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    statedic, mdp = env.getMDP()

    # Initialisation de V
    LV = []
    states = [k for (k, val) in statedic.items()]
    init_state = [k for (k, val) in statedic.items() if val == 0][0]

    for state in states:
        val = int(init_state[state.index('2')])
        if val == 0:
            LV.append(-0.001)
        if val == 2:
            LV.append(-0.001)
        if val == 3:
            LV.append(1)
        if val == 4:
            LV.append(1)
        if val == 5:
            LV.append(-1)
        if val == 6:
            LV.append(-1)

    # Value Iteration
    states = [k for (k, val) in mdp.items()]
    gamma = 0.9
    i = 0
    epsilon = 10e-5
    old_LV = [10] * len(statedic)
    Pol = [0] * len(statedic)
    LVtemp = LV.copy()

    while np.linalg.norm(np.array(LV)-np.array(old_LV)) >= epsilon:
        for state in states:
            Vi = -99999999999
            actions = mdp.get(state)
            state_index = statedic.get(state)
            for j in range(len(actions)):
                Via = 0
                action = actions.get(j)
                for k in range(len(action)):
                    Vs = action[k][1]
                    Vindex = statedic.get(Vs)
                    V = LV[Vindex]
                    Via += action[k][0] * (0 + gamma*V)
                if Via > Vi:
                    Vi = Via
                    Pol_i = action

            old_LV = LV.copy()
            LVtemp[state_index] = Vi
            Pol[state_index] = Pol_i

        LV = LVtemp.copy()
        i += 1

    #print(i)
    #print(LV)
    #print(statedic)
    #print(Pol)

    return Pol

print(ValueIteration(0))