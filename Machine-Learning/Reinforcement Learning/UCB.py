import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
#importing the dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv") 
#this is the data set for simulation
#basically as of now we dont have any data.
#we have 10 versions of the same ad,each time user connects to account
#we put one of these ads onto the website
#if the user clicks we give reward as 1 else zero
# the decision we make at nth round will depend on the observations
# that are known about the previous n-1 rounds
# the 1s are the user will click on that version of the ad

#----------------Implementing UCB from scratch--------------
N = 10000 #number of rounds
d = 10 # number of variation of ads
numbers_of_selections = [0]*d 
#create a vector of size d of only zero
sums_of_rewards = [0]*d
ads_selected  = []
total_reward = 0
for n in range(0,N):
    i = 0
    max_upper_bound = 0
    for i in range(0,d):
        if numbers_of_selections[i] > 0 :
            average_reward = sums_of_rewards[i]/ numbers_of_selections[i]
            #now we need upper confidence bound
            delta_i = math.sqrt(1.5*(math.log(n+1))/numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else :# dduring first 10 rounds we will just follow the ads one after the other
#so we get somee info on reward and all
            upper_bound = 1e400  # this is 10 ki power 400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad =  i
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
    #now we check the total reward by using UCB
    
# visualizations through a histogram
plt.hist(ads_selected)
plt.title("Histogram of ads")
plt.xlabel("ads")
plt.ylabel("Number of times each ad is selected")
plt.show()