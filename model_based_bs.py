# Computational Cognitive Neuroscience - Harvard University
# Gershman Lab
# Author: Rohan Kalra
# Date: August 2019

#############################################################################################

# Import necessary packages

import sys
import time
import math
import random
import numpy as np
import pickle as pk
import seaborn as sns
import matplotlib.pyplot as plt

#############################################################################################

# Set global variables
list_of_observations = list(range(0,11))
stop_locations = list()
keys = list(range(0,21))
rpelist = {key:list() for key in keys}

# Environment specific information
nStates = 20
nActions = 2
nObservations = 10

# Agent specific information
aLearningRate = .1
aDiscount = .95
aExplorationRate = .1
aTrials = 10000

# Other information
ITI = 0
d = {0:'RUN', 1:'STOP'}
RUN = 0
STOP = 1
state_to_observation = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:1, 12:2, 13:3, 14:4, 15:5, 16:6, 17:7, 18:8, 19:9, 20:10}

#############################################################################################

# Class for RL environment

class Environment:
    def __init__(self):
        self.state = ITI  # Default State (ITI)
        self.observation = state_to_observation[self.state] # Default observation (ITI)
        self.top_track = True # Select default track to perform trial on
        self.location = -999 # D

    def determine_type_of_track(self):
        random_number = np.random.random()
        if random_number > 0.2:
            self.top_track = True
        else:
            self.top_track = False 

    def determine_reward_location(self):
        if self.top_track == True:
            self.location = np.random.randint(low=3, high=8)
        else:
            self.location = -999

    def take_action(self, action):
        if self.state == ITI:
            random_number = np.random.random()
            if random_number > .1:
                reward = 0
                self.state = ITI
            else:
                if self.top_track == True:
                    self.state = 1
                    reward = 0
                else:
                    self.state = 11
                    reward = 0
        else:
            if self.top_track == True:
                if action == STOP:
                    # STOP: go back to the beginning, get no reward
                    reward = 0
                    self.state = ITI
                else:
                    # RUN: keep running on track
                    if self.state < (nStates/2):
                        self.state += 1
                        if self.state == self.location:
                            reward = 1
                            self.state = ITI
                        else:
                            assert(self.state < self.location)
                            reward = 0
                    else:
                        reward = 0
                        self.state = ITI
            else:
                if action == STOP:
                    # STOP: go back to the beginning, get no reward
                    reward = 0
                    self.state = ITI
                else:
                    # RUN: keep running on track
                    if self.state < nStates:
                        self.state += 1
                        reward = 0
                    else:
                        reward = 0
                        self.state = ITI 
        self.observation = state_to_observation[self.state]
        return self.state, self.observation, reward

#############################################################################################

# Class for RL Agent

class Agent:
    def __init__(self, learning_rate=aLearningRate, discount=aDiscount, exploration_rate=aExplorationRate, trials=aTrials):
        self.learning_rate = aLearningRate # How much we appreciate new q-value over current
        self.discount = aDiscount # How much we appreciate future reward over current
        self.exploration_rate = aExplorationRate # Initial exploration rate
        self.trials = aTrials # Number of Trials per learning episode
        self.q_table = np.zeros((nActions, nStates+1)) # Spreadsheet (Q-table) for rewards accounting
        self.belief_state = np.zeros((1, nStates+1))
        self.belief_state[0][ITI] = 1
        # self.belief_state[0][1] = .8
        # self.belief_state[0][11] = .2

        with open('transition_probability.pkl', 'rb') as f:
            self.P = pk.load(f)        

    def get_next_action(self, belief_state):
        if random.random() > self.exploration_rate: # Explore (gamble) or exploit (greedy)
            return self.greedy_action(belief_state)
        else:
            return self.random_action()

    def greedy_action(self, belief_state):
        belief_q = np.dot(self.q_table, belief_state.T)
        return belief_q.argmax()

    def random_action(self):
        return RUN if random.random() < 0.5 else STOP

    def update_belief_state(self, new_observation, action):
        new_belief_state = np.zeros((1, nStates+1))
        old_belief_state = self.belief_state
        for new_state in range(0, nStates+1):
            for old_state in range(0, nStates+1):
                to_add = old_belief_state[0][old_state] * self.P[old_state][action][new_state][new_observation]
                new_belief_state[0][new_state] += to_add
        s = new_belief_state[0].sum()
        new_belief_state[0] = new_belief_state[0] / s
        self.belief_state = new_belief_state
        return new_belief_state

    def update(self, old_belief_state, new_belief_state, action, reward, env_location, env_state):
        old_belief_q = np.dot(self.q_table, old_belief_state.T)
        old_value = old_belief_q[action]
        future_action = self.greedy_action(new_belief_state)
        new_belief_q = np.dot(self.q_table, new_belief_state.T)
        future_reward = new_belief_q[future_action]
        for old_state in range(0, nStates+1):
            rpe = reward + self.discount * future_reward - old_value
            rpelist[env_state].append(rpe[0])
            # if reward == 1:
            #     rpelist[env_location].append(rpe[0])
            self.q_table[action][old_state] += (self.learning_rate * old_belief_state[0][old_state] * (rpe))

#############################################################################################

def calculate_transition_probabilities(iterations=100):

    # breakpoint()

    start = time.time()

    agent = Agent()
    env = Environment()

    count = np.zeros((nStates+1, nActions, nStates+1, nObservations+1))
    count_marginal = np.zeros((nStates+1, nActions))
    P = np.zeros((nStates+1, nActions, nStates+1, nObservations+1))

    for old_state in range(0, nStates+1):
        for action in range(0, nActions):
            for iteration in range(0, iterations):
                if old_state == 0:
                    env.determine_type_of_track()
                elif old_state in range(1, 11):
                    env.top_track = True
                else:
                    env.top_track = False
                env.determine_reward_location()
                if env.top_track and old_state >= env.location:
                    continue
                count_marginal[old_state][action] += 1
                env.state = old_state
                new_state, new_observation, reward = env.take_action(action)
                if old_state == 5 and action == 0:
                    print(new_state, new_observation)
                count[old_state][action][new_state][new_observation] += 1
            for new_state in range(0, nStates+1):
                for new_observation in range(0, nObservations+1):
                    if count_marginal[old_state][action] == 0:
                        P[old_state][action][new_state][new_observation] = 0
                    else:
                        P[old_state][action][new_state][new_observation] = count[old_state][action][new_state][new_observation] / count_marginal[old_state][action]
    p1 = 1
    p2 = 1
    for i in range(1, 8):
        print(i, P[i][0][i+1][i+1])
        p1 *= P[i][0][i+1][i+1]
        p2 *= P[i+10][0][i+11][i+1]
        print(i, ' tr1 ', p1, '  tr2 ', p2, ' post wrong ', p1 / (p1 + p2), ' post corr', p1 * 0.8 /(p1 * 0.8 + p2 * 0.2)) 

    end = time.time()

    # print(P[0][0][1][1])
    # print(P[0][0][11][1])
    # print(P[0][0][0][0])

    with open('transition_probability.pkl', 'wb') as f:
        pk.dump(P, f)

    print('Done calculating transition probabilities! Took {} seconds!'.format(end-start))

#############################################################################################

# Misc. Methods

def line_graph(data):
    plt.figure(figsize=(15,15))
    x_val = [x[0] for x in data]
    y_val = [x[1] for x in data]
    plt.plot(x_val,y_val)
    plt.plot(x_val,y_val,'or')
    plt.show()   

def bar_graph(labels, values1, values2):
    plt.figure(figsize=(15,15))
    # set width of bar
    barWidth = 0.25
     
    # set height of bar
    bars1 = values1
    bars2 = values2
     
    # Set position of bar on X axis
    r1 = np.arange(len(values1))
    r2 = [x + barWidth for x in r1]
     
    # Make the plot
    plt.bar(r1, bars1, color='#008000', width=barWidth, edgecolor='white', label='RUN')
    plt.bar(r2, bars2, color='#FF0000', width=barWidth, edgecolor='white', label='STOP')
     
    # Add xticks on the middle of the group bars
    plt.xlabel('State', fontweight='bold')
    plt.title('Probability of Each Action at Each State')

    # Create legend & Show graphic
    plt.legend()
    plt.show()

def convert_q2p(l, beta):
    to_ret = list()
    for pair in l:
        run = math.exp(pair[0]*beta)
        stop = math.exp(pair[1]*beta)
        s = run + stop
        run_p = run/s
        stop_p = stop/s
        to_ret.append((run_p, stop_p))
    return to_ret

#############################################################################################

def main():
    
    # breakpoint()
    prob_bottom = [0]

    sns.set(color_codes=True)

    calculate_transition_probabilities(iterations=1000)

    # setup simulation
    agent = Agent()
    print('STARTING BELIEF STATE:', agent.belief_state)
    heatmap = agent.belief_state
    print()

    env = Environment()
    env.determine_type_of_track()
    env.determine_reward_location()

    total_reward = 0 # Score keeping
    reward_over_time = list()
    locations = []

    # main loop
    count = 0
    while count < agent.trials:

        """Track locations to ensure they are drawn from 
        Gaussian distribution"""
        if env.location != -999:
            locations.append(env.location)

        old_state = env.state
        old_belief_state = agent.belief_state
        action = agent.get_next_action(old_belief_state)
        if old_state != ITI and action == STOP and env.location == -999:
            stop_locations.append(old_state)
        new_state, new_observation, reward = env.take_action(action)
        new_belief_state = agent.update_belief_state(new_observation, action)
        # print(old_state, action, new_state, new_observation, reward)
        print(new_belief_state)
        
        if new_observation != 0:
            prob_bottom.append(np.sum(new_belief_state[0][11:]))
            heatmap = np.concatenate((heatmap, new_belief_state), axis=0)
        to_use = new_state
        agent.update(old_belief_state, new_belief_state, action, reward, env.location, to_use)

        if old_state != ITI and new_state == ITI:
            count += 1
            print('Trial {} finished!'.format(count))
            print('BELIEF STATE AFTER {} TRIALS: {}'.format(count, agent.belief_state))
            print()
            env.determine_type_of_track()
            env.determine_reward_location()
            prob_bottom += list([1,1,1,1])


        total_reward += reward
        reward_over_time.append((count, total_reward))

    # sns.heatmap(heatmap, cmap="YlGnBu")
    states = []
    rpes = []
    for state, l in rpelist.items():
        if l == []:
            states.append(state)
            rpes.append(float(0))
            continue
        states.append(state)
        rpes.append(sum(l)/len(l))
    # print(states, rpes)
    # sns.barplot(x=states, y=rpes)
    print("MEDIAN {}".format(np.median(stop_locations)))
    sns.distplot(stop_locations, bins=20, kde=True);
    print(list(range(0,11)), prob_bottom)
    # plt.bar(x=list(range(0,11)), height=prob_bottom)

    plt.figure(figsize=(15,15))
    plt.hist(locations, bins='auto')
    
    # Graph the Results

    run_qvalues = agent.q_table[0,:].tolist()
    stop_qvalues = agent.q_table[1,:].tolist()
    l = list(zip(run_qvalues, stop_qvalues))
    l = convert_q2p(l, 100)
    res = [[ i for i, j in l ], 
       [ j for i, j in l ]]
    run_p_values = res[0]
    stop_p_values = res[1]

    line_graph(reward_over_time)
    list_of_states = list(range(nStates))
    bar_graph(list_of_states, run_p_values, stop_p_values)

#############################################################################################

if __name__ == "__main__":
    print()
    main()
    print()

#############################################################################################

# END OF FILE
