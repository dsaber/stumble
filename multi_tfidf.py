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
from sklearn.linear_model import LogisticRegression

import json 


def make_xy(df):

	Y = df["label"]
	X = df.drop("label", axis=1)
	return X, Y

def domain_extractor(ser):

	pass

def text_extractor(X):

	all_docs = {"title": [], "url": [], "body": [], "related": []} 
	keys = ["title", "url", "body", "related"]

	for i in range(X.shape[0]):
		print i 
		json_obj = json.loads(X["boilerplate"][i])
		for key in keys:
			try: 
				if json_obj[key] is not None:
					if type(json_obj[key]) == list: 
						all_docs[key].append(' '.join(json_obj[key]))
					else:
						all_docs[key].append(json_obj[key])
				else:
					all_docs[key].append(" ")
			except:
				all_docs[key].append(" ")
	return all_docs 




if __name__ == "__main__":

	df = pd.read_csv("data/train.tsv", sep="\t")
	X, Y = make_xy(df)
	all_docs = text_extractor(X)
	print all_docs.keys()

	# make a bag of words
	cv = TfidfVectorizer(stop_words="english")
	cv_mat_title = cv.fit_transform(all_docs["title"])
	# vocab_df_title = pd.DataFrame(cv_mat_title.toarray(), columns=cv.get_feature_names())

	cv = TfidfVectorizer(stop_words="english")
	cv_mat_url = cv.fit_transform(all_docs["url"])
	# vocab_df_url = pd.DataFrame(cv_mat_url.toarray(), columns=cv.get_feature_names())

	cv = TfidfVectorizer(stop_words="english")
	cv_mat_body = cv.fit_transform(all_docs["body"])
	# vocab_df_body = pd.DataFrame(cv_mat_body.toarray(), columns=cv.get_feature_names())

	# dropping columns unecessary for analysis 
	X = X.drop("boilerplate", axis=1)
	X = X.drop("url", axis=1)
	X = X.drop("urlid", axis=1)

	# # join our X with our bag of words
	# print X.shape
	# X = X.join(vocab_df, how="inner")
	# print X.shape


	cv_mat = hstack([cv_mat_body, cv_mat_url, cv_mat_title])

	nb = MultinomialNB()
	print "Running Multinomial Naive Bayes"
	scores = cross_validation.cross_val_score(nb, cv_mat, Y, cv=5, scoring="roc_auc", verbose=5)
	print "Accuracy: " + str(np.mean(scores))

	nb.fit(cv_mat, Y)
	pred = nb.predict(cv_mat)


	X["pred"] = pred
	safe_cols = [] 
	for col in X.columns:
		if type(X[col][0]) == np.float_ or type(X[col][0]) == np.int_:
			safe_cols.append(col)
	print safe_cols 
	sv = LogisticRegression() 
	scores = cross_validation.cross_val_score(sv, X[safe_cols], Y, cv=5, scoring="roc_auc", verbose=5)
	print "Accuracy: " + str(np.mean(scores))


	# nb = MultinomialNB()
	# print "Running Multinomial Naive Bayes"
	# scores = cross_validation.cross_val_score(nb, cv_mat_url, Y, cv=5, scoring="roc_auc", verbose=5)
	# print "Accuracy: " + str(np.mean(scores))

	# nb = MultinomialNB()
	# print "Running Multinomial Naive Bayes"
	# scores = cross_validation.cross_val_score(nb, cv_mat_title, Y, cv=5, scoring="roc_auc", verbose=5)
	# print "Accuracy: " + str(np.mean(scores))
	







