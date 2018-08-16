#Natural Language Processing

#Importing the dataset
#Ignore the quotes in the text
#To prevent the reviews from being identified as factors
dataset_original = read.delim('Restaurant_Reviews.tsv', quote='', stringsAsFactors = FALSE)

#Cleaning the texts
#install.packages('tm')
#install.packages('SnowballC')
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_original$Review))

#Converting all words to lowercase to have one unique version of the word
corpus = tm_map(corpus, content_transformer(tolower))
#Removing the numbers from the reviews
corpus = tm_map(corpus, removeNumbers)
#Remove punctuation to avoid having a separate column for each punctuation mark
corpus = tm_map(corpus, removePunctuation)
#Remove irrelavant words
corpus = tm_map(corpus, removeWords, stopwords())
#Stemming Words
corpus = tm_map(corpus, stemDocument)
#Removing extra white spaces
corpus = tm_map(corpus, stripWhitespace)

#Creating Bag of Words Model
dtm = DocumentTermMatrix(corpus)
#Filter words that appear only once and not frequently in the review
dtm = removeSparseTerms(dtm, 0.999)
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked

#You need to factor 0,1 because otheriwse R would treat them as numeric. We want R to interpret it as characters.
# Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

#Random Forest Classififer
# Splitting the dataset into the Training set and Test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.80)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Random Forest Classification to the Training set
# install.packages('randomForest')
library(randomForest)
set.seed(123)
classifier = randomForest(x = training_set[-692],
                          y = training_set$Liked,
                          ntree = 10)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)
