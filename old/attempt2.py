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
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import scale

import json 


def make_dummy(df, column):

	dum_dum = pd.get_dummies(df[column])
	for col in dum_dum.columns:
		df[col] = dum_dum[col]
	temp = df.drop(column, axis=1)
	return temp 


def make_xy(df):

	Y = df["label"]
	X = df.drop("label", axis=1)
	return X, Y


def text_extractor(X):

	all_docs = {"title": [], "url": [], "body": [], "related": []} 
	keys = ["title", "url", "body", "related"]

	for i in range(X.shape[0]):
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

	df = pd.read_csv("data/train_url.tsv", sep="\t")
	X, Y = make_xy(df)
	all_docs = text_extractor(X)

	X_test = pd.read_csv("data/test_url.tsv", sep="\t")
	all_test_docs = text_extractor(X_test)



	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 	
	# TF-IDF's for TRAINING SET
	cv_title = TfidfVectorizer(stop_words="english")
	cv_title.fit(all_docs["title"])
	cv_mat_title = cv_title.transform(all_docs["title"])
	# svd_title = TruncatedSVD(n_components=10)
	# cv_mat_title = scale(svd_title.fit_transform(cv_mat_title))

	cv_url = TfidfVectorizer(stop_words="english")
	cv_url.fit(all_docs["url"])
	cv_mat_url = cv_url.transform(all_docs["url"])
	# svd_url = TruncatedSVD(n_components=10)
	# cv_mat_url = scale(svd_url.fit_transform(cv_mat_url))

	cv_body = TfidfVectorizer(stop_words="english")
	cv_body.fit(all_docs["body"])
	cv_mat_body = cv_body.transform(all_docs["body"])
	# svd_body = TruncatedSVD(n_components=100)
	# cv_mat_body = scale(svd_body.fit_transform(cv_mat_body))

	# TF-IDF's for TEST SET
	test_title = cv_title.transform(all_test_docs["title"])
	test_url = cv_url.transform(all_test_docs["url"])
	test_body = cv_body.transform(all_test_docs["body"])
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


	# dropping columns unecessary for analysis 
	# TRAINING
	X = X.drop("boilerplate", axis=1)
	X = X.drop("url", axis=1)
	X = X.drop("urlid", axis=1)

	# TEST
	X_test = X_test.drop("boilerplate", axis=1)
	X_test = X_test.drop("url", axis=1)
	urlid = X_test["urlid"]
	X_test = X_test.drop("urlid", axis=1)
	print urlid

	# get ready for Naive Bayes
	cv_mat = hstack([cv_mat_body, cv_mat_url, cv_mat_title])
	cv_mat_test = hstack([test_body, test_url, test_title])

	print cv_mat.shape
	print cv_mat_test.shape 
	print cv_mat 

	# Naive Bayes Testing 
	nb = MultinomialNB()
	print "Running Multinomial Naive Bayes"
	scores = cross_validation.cross_val_score(nb, cv_mat, Y, cv=5, scoring="roc_auc", verbose=5)
	print "Accuracy: " + str(np.mean(scores))

	# Production model 
	nb_production = MultinomialNB()
	nb_production.fit(cv_mat, Y)
	pred = nb_production.predict(cv_mat)
	X["pred"] = pred

	pred_test = nb_production.predict(cv_mat_test)
	X_test["pred"] = pred_test






	# Find columns we can actually use without problems
	safe_cols = [] 
	unsafe_cols = [] 
	for col in X.columns:
		if type(X[col][0]) == np.float_ or type(X[col][0]) == np.int_:
			safe_cols.append(col)
		else:
			unsafe_cols.append(col)

	sv = RandomForestClassifier(n_estimators=500, n_jobs=-1)
	sv.fit(X[safe_cols], Y)
	# scores = cross_validation.cross_val_score(sv, X[safe_cols], Y, cv=10, scoring="roc_auc", verbose=5)
	# print "Accuracy: " + str(np.mean(scores))
	nb = MultinomialNB() 
	nb.fit(cv_mat, Y)
	logreg = LogisticRegression() 
	logreg.fit(cv_mat, Y)


	nb_pred = nb.predict(cv_mat_test)
	sv_pred = sv.predict(X_test[safe_cols])
	log_pred = logreg.predict(cv_mat_test)

	final_predictions = pd.DataFrame(urlid)
	final_predictions["label"] = nb_pred + sv_pred + log_pred
	final_predictions["label"][final_predictions["label"] == 1] = 0
	final_predictions["label"][final_predictions["label"] == 2] = 1
	final_predictions["label"][final_predictions["label"] == 3] = 1
	print final_predictions
	final_predictions.to_csv("predictions3.csv", index=False) 





