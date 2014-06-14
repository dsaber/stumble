import pandas as pd 
import numpy as np 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation
from sklearn import metrics
from scipy.sparse import vstack 
from scipy.sparse import hstack 
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix 

import json 


def make_xy(df):

	Y = df["label"]
	X = df.drop("label", axis=1)
	return X, Y

def domain_extractor(ser):

	pass

def text_extractor(X):

	all_docs = [] 

	for i in range(X.shape[0]):
		str_array = []
		json_obj = json.loads(X["boilerplate"][i])
		for key, val in json_obj.iteritems():
			if val is not None: 
				str_array.append(val.encode("ascii", "ignore"))
		all_docs.append(' '.join(str_array))

	return all_docs 




if __name__ == "__main__":

	df = pd.read_csv("data/train.tsv", sep="\t")
	X, Y = make_xy(df)
	all_docs = text_extractor(X)

	# make a bag of words
	cv = TfidfVectorizer(stop_words="english")
	cv_mat = cv.fit_transform(all_docs)
	vocab_df = pd.DataFrame(cv_mat.toarray(), columns=cv.get_feature_names())

	# dropping columns unecessary for analysis 
	X = X.drop("boilerplate", axis=1)
	X = X.drop("url", axis=1)
	X = X.drop("urlid", axis=1)

	alc_sparse = np.array(X["alchemy_category"])
	alc_sparse = coo_matrix(alc_sparse)
	alc_sparse = alc_sparse.transpose(copy=True) 
	print alc_sparse

	print cv_mat.shape
	print alc_sparse.shape

	cv_mat = hstack([cv_mat, alc_sparse])

	# # join our X with our bag of words
	# print X.shape
	# X = X.join(vocab_df, how="inner")
	# print X.shape 

	nb = MultinomialNB()
	print "Running Multinomial Naive Bayes"
	scores = cross_validation.cross_val_score(nb, cv_mat, Y, cv=5, scoring="roc_auc", verbose=5)
	print "Accuracy: " + str(np.mean(scores))


	







