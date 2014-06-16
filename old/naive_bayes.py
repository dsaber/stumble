import pandas as pd 
import numpy as np 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation
from sklearn import metrics

import json 

if __name__ == "__main__":

	X_train = pd.read_csv("data/X_process.csv")
	Y_train = pd.read_csv("data/Y.csv")

	nb = MultinomialNB() 
	scores = cross_validation.cross_val_score(nb, X_train, Y_train, cv=5, scoring="accuracy")
	print "Accuracy: " + np.mean(scores) 