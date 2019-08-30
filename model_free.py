

#------------------------------------------------------------------------------------------#

# CCNL
# Rohan Kalra, Summer 2019
# Markov Decision Process 1

#------------------------------------------------------------------------------------------#

# Import necessary packages

import math
import random
from pyforest import * # Includes all packages useful for data science

#------------------------------------------------------------------------------------------#

# Global Variables

d = {0:'RUN', 1:'STOP'}
RUN = 0
STOP = 1

nActions = 2
nStates = 10
ITI = 0

#------------------------------------------------------------------------------------------#

# Class for RL environment

class Environment:

    def __init__(self, length=nStates):
        self.length = length # Length of the track
        self.state = ITI  # Start at the pre-trial state
        self.track = 2 # Select default track to perform trial on (Track 1 has 100% reward rate, Track 2 has 80% reward rate)

    # This method will be useful in the future, when both tracks must be experienced within the same learning episode
    def determine_track(self):
        rn = np.random.random()
        if rn > .3:
            self.track = 1
        else:
            self.track = 2

    # This method determines reward location based on current track as specified by environment
    def determine_reward_location(self):
        if self.track == 1:
            self.location = np.random.randint(low=3, high=8)
        else: 
            rn = np.random.random()
            if rn > 0.2: 
                # self.location = int(np.random.normal(loc=3, scale=2)) # Gaussian Distribution: Centered at 3, Std. of 2
                self.location = np.random.randint(low=3, high=8)
            else:
                self.location = -999
        return self.location

    # This method carries out an action on behalf of the agent and updates environment information, 'action' is the action to be carried out
    def take_action(self, action):
        if self.state == ITI:
            rn = np.random.random()
            if rn > .1:
                # Trial doesn't begin, go back to ITI
                reward = 0
                self.state = ITI
            else:
                # Trial starts
                self.state = 1
                reward = 0
        elif action == STOP:
            # STOP: go back to the beginning, get no reward
            reward = 0
            self.state = ITI
        elif action == RUN:
            # RUN: keep running on track
            if self.state < self.length:
                self.state += 1
                if self.state == self.location:
                    reward = 1
                    self.state = ITI
                else:
                    reward = 0
            else:
                reward = 0
                self.state = ITI
        return self.state, reward

#------------------------------------------------------------------------------------------#

# Class for RL Agent

class Agent:
    def __init__(self, learning_rate=0.1, discount=0.95, exploration_rate=0.1, trials=10000):
        self.q_table = np.zeros((nActions, nStates+1)) # Spreadsheet (Q-table) for rewards accounting
        self.learning_rate = learning_rate # How much we appreciate new q-value over current
        self.discount = discount # How much we appreciate future reward over current
        self.exploration_rate = exploration_rate # Initial exploration rate
        self.trials = trials # Number of Trials per learning episode

    def get_next_action(self, state):
        if random.random() > self.exploration_rate: # Explore (gamble) or exploit (greedy)
            return self.greedy_action(state)
        else:
            return self.random_action()

    def greedy_action(self, state):
        # Is RUN reward bigger?
        if self.q_table[RUN][state] > self.q_table[STOP][state]:
            return RUN
        # Is STOP reward bigger?
        elif self.q_table[STOP][state] > self.q_table[RUN][state]:
            return STOP
        # Rewards are equal, take random action
        return RUN if random.random() < 0.5 else STOP

    def random_action(self):
        return RUN if random.random() < 0.5 else STOP

    def update(self, old_state, new_state, action, reward):
        # Old Q-table value
        old_value = self.q_table[action][old_state]
        # Select next best action...
        future_action = self.greedy_action(new_state)
        # What is reward for the best next action?
        future_reward = self.q_table[future_action][new_state]

        # Main Q-table updating algorithm
        new_value = old_value + self.learning_rate * (reward + self.discount * future_reward - old_value)
        self.q_table[action][old_state] = new_value

#------------------------------------------------------------------------------------------#

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

# Converts q-values to softmax probabilities: 'l' is the list of tuples of q-values to be converted, 'beta' is the beta value to be used in softmax
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

#------------------------------------------------------------------------------------------#

# Main method

def main():

    # The following variables will help in tracking values for data analysis/graphing
    observations = list(range(0,11)) # 0-10
    states_visited = {key:0 for key in observations}
    run_qvalues = list()
    stop_qvalues = list()
    stop_distances = list()

    # setup simulation
    agent = Agent()
    env = Environment()

    total_reward = 0 # Score keeping
    reward_over_time = list()
    locations = []

    count = 0
    # main loop
    env.determine_reward_location()
    while count < agent.trials:

        """Ensure locations are drawn from Gaussian distribution"""
        if env.location != -999:
            locations.append(env.location)

        old_state = env.state
        states_visited[old_state] += 1
        action = agent.get_next_action(old_state)
        if action == STOP and old_state != ITI and env.location == -999:
            stop_distances.append(old_state)
        new_state, reward = env.take_action(action)
        if old_state != ITI and new_state == ITI:
            count += 1
            print('Trial {} finished..'.format(count))
            env.determine_reward_location()
            # env.determine_track()
        agent.update(old_state, new_state, action, reward)
        print(count)
        print(old_state, new_state, action, env.location)
        # print(agent.q_table)
        # time.sleep(.1)
        total_reward += reward
        reward_over_time.append((count, total_reward))

    plt.figure(figsize=(15,15))
    print(states_visited)
    # sns.distplot(states_visited, bins=10, kde=True)
    # sns.distplot(stop_distances, bins=10, kde=True);
    print('MEDIAN {}'.format(np.median(stop_distances)))
    plt.figure(figsize=(15,15))
    plt.hist(locations, bins='auto')
    
    # Graph the Results
    sns.set(color_codes=True)
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
    bar_graph(list_of_states, run_qvalues, stop_qvalues)
    
#------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    main()

#------------------------------------------------------------------------------------------#

