#Apriori Algorithm

#Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

#Apriori algorithm expects here a list of lists as input
#One big list containing all the transactions and list for each transaction inside it
#Apriori algorithm uses values as strings

transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

#Visualizing the results
result = list(rules)

results_list = []
for i in range(0, len(result)):
    results_list.append('RULE:\t' + str(result[i][0]) + '\nSUPPORT:\t' + str(result[i][1]))
    