#Eclat Model

#Data Preprocessing
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)

#arules package used for Apriori Algorithm takes input as Sparse Matrix
#install.packages("arules")
library(arules)

#Apriori Algorithm should be trained on dataset which doesn't have duplicates
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',',
                            rm.duplicates = TRUE)
#5 transactions having 1 dupicate each
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

#Training Apriori on the dataset
#Tune the parameters of support and confidence to get optimal rules
#Support = 3 * 7 / 7500 and Confidence = 0.8 -> 0.8/2
rules = eclat(data = dataset,
                parameter = list(support = 0.004, minlen = 2))
inspect(sort(rules, by = 'support')[1:10])