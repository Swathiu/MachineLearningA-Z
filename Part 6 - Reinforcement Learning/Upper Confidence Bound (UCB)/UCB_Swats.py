#Upper Confidence Bound

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

#Importing Dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
#No independant and dependant variables here in this case - It is
#the key difference between current problem and previous problems

#Strategy of the algorithm is dynamic. It depends from the beginning 
#of the experiment upto the present time. The current result depends on the
# results of the previous round results.
#Hence called reinforcement learning is also called interactive learning
#or Online Learning
#For random selection implementation (random_selection.py), the total reward is around 1200 points

#Implementing UCB
N = 10000
d =  10
#Initializes a vector of length d with 0s
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
ads_selected = []
total_reward = 0

for n in range(0,N):
    max_upper_bound = 0
    for i in range(0,d):
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i]/numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n,ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
    
#Visualizing results
plt.hist(ads_selected, edgecolor='black', linewidth=1.2)
plt.title('Histogram of Ads Selections')
plt.xlabel('Ads')
plt.ylabel('Frequency of Ads')
plt.show()