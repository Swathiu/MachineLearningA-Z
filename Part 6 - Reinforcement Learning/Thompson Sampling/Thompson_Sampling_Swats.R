#Thompson Sampling

#Importing dataset
dataset = read.csv('Ads_CTR_Optimisation.csv')

#Implementing UCB
#Creating vector of size d
number_of_rewards_0 = integer(d)
number_of_rewards_1 = integer(d)
sums_of_rewards = integer(d)
ads_selected = integer()
total_reward = 0
N = 10000
d = 10

for (n in 1:N) {
  ad = 0
  max_random_number = 0
  for (i in 1:10){
      random_number = rbeta(n= 1,
                            shape1 = number_of_rewards_1[i] + 1, 
                            shape2 = number_of_rewards_0[i] + 1)
      if (random_number > max_random_number){
        max_random_number = random_number
        ad = i
      }
  }
  ads_selected = append(ads_selected, ad)
  reward = dataset[n, ad]
  if (reward == 1){
    number_of_rewards_1[ad] = number_of_rewards_1[ad] + 1
  }
  else{
    number_of_rewards_0[ad] = number_of_rewards_0[ad] + 1
  }
  total_reward = total_reward + reward
}


# Visualising the results
hist(ads_selected,
     col = 'blue',
     main = 'Histogram of Ads selections',
     xlab = 'Ads',
     ylab = 'Frequency of Ad')
