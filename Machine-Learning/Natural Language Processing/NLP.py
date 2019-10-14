#importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv",delimiter = "\t", quoting = 3)
#quoting = 3 will just ignore the double quotes in the reviews 
#in csv the delimiter is , whereas in tsv the delimiter is tab(space)
#we use tsv here becuase there are already , (duplicate delimiters)in the review itself

#-----next step in NLP is cleaning the text
#means we will only consider the relevant words and get rid of things like punctuation
import re #re is  regular expression library
import nltk #natural language processing kit
nltk.download("stopwords") # list of words which are irrelevant
from nltk.corpus import stopwords
#if in our review we have certain irrelevant words
#if those words are in STOPWORDS we will remove them from the review
#----------------stemming---------------------
from nltk.stem.porter import PorterStemmer
#we apply stemming so that we dont have way too many words in the end
# example loved,loves,loving all will be converted to love
# we only need the root of the word
# ----------------------This is the cleaning for the first review do the same 
#for all the reviews
"""
review = re.sub("[^a-z A-Z]"," ",dataset["Review"][0])
#removing everything apart from a-z and A-Z where ^ denotes the complement
#re.sub is the method used for carrying out this removing process
#removing numbers and punctuation marks
review = review.lower()
#put all the words into lower case
 # here word LOVED tells us that review is positive but word
 # this has no relevance so we need to remove words which are
 # irrelevant
review = review.split() #making the string into the list
ps = PorterStemmer() # object of the stemming class 
review = [ ps.stem(word) for word in review if not word in set(stopwords.words("english"))     ] 
# this removed the irrelevant word THIS
#and stem change the loved to love
review = " ".join(review) #join the list back into the string
"""
corpus = [] #all the edited reviews
for i in range(1000):
    review = re.sub("[^A-Z a-z]", " ", dataset["Review"][i] )
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words("english"))]
    review = " ".join(review)
    corpus.append(review)
    
# creating bag of words model
    # a matrix containing a lot of zeros is called sparse matrix
    #and this feature is known as sparsity because here in the matrix
    #of the words that is formed is likely to remain empty for individual
    # review, so we try to reduce sparsity as much as possible
    # TOKENIZATION is taking all the words in the review we will allocate a column for each word
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500 )
#with  max features we only take the most frequent words
# all the things we did manually we can do here only # here we can use stop words parameter of this class
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,[1]].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
