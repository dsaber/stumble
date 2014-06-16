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


def tfidf_and_svd(docs, test_docs, num, svd=True, min_df=3, ngram_range=(1,2)):
	
	cv = TfidfVectorizer(stop_words="english", min_df=min_df, ngram_range=ngram_range)
	cv.fit(list(docs) + list(test_docs))
	cv_mat = cv.transform(docs)
	cv_mat_test = cv.transform(test_docs)

	if svd is False:
		return cv_mat, cv_mat_test

	svd = TruncatedSVD(n_components=num)
	doc_result = svd.fit_transform(cv_mat)
	test_doc_result = svd.transform(cv_mat_test)

	return doc_result, test_doc_result








if __name__ == "__main__":

	df = pd.read_csv("data/train_url.tsv", sep="\t")
	X, Y = make_xy(df)
	all_docs = text_extractor(X)

	X_test = pd.read_csv("data/test_url.tsv", sep="\t")
	all_test_docs = text_extractor(X_test)


	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 	
	# TF-IDF's for TRAINING SET
	cv_mat_title, test_title = tfidf_and_svd(all_docs["title"], all_test_docs["title"], 10, False)
	cv_mat_url, test_url = tfidf_and_svd(all_docs["url"], all_test_docs["url"], 10, False)
	cv_mat_body, test_body = tfidf_and_svd(all_docs["body"], all_test_docs["body"], 100, False) 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


	# Finalizing feature set
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


	# Find columns we can actually use without problems
	safe_cols = ["insidershealth", "sportsillustrated", "huffingtonpost", "allrecipes",
				 "bleacherreport", "blogs", "ivillage", "foodnetwork", "blog", "dailymail",
				 "epicurious", "womansday", "bbc", "guardian", "marthastewart", "itechfuture",
				 "popsci", "collegehumor", "news", "buzzfeed", "wimp", "youtube", "telegraph",
				 "npr", "gizmodo", "menshealth", "seriouseats", "instructables", "smittenkitchen",
				 "thepioneerwoman", "geek", "cbsnews", "pred"] 
	# get text features finalized
	training_features = hstack([cv_mat_title, cv_mat_url, cv_mat_body])
	test_features = hstack([test_title, test_url, test_body])

	# Intermediate Model
	print "Training Intermediate Logistic Regression"
	logreg = LogisticRegression(penalty="l2", dual=True, tol=0.0001)
	logreg.fit(training_features, Y)
	X["pred"] = logreg.predict_proba(training_features)[:, 1]
	X_test["pred"] = logreg.predict_proba(test_features)[:, 1]



	# WRAP IT UP # 
	final_predictions = pd.DataFrame(urlid)
	final_predictions["label"] = X_test["pred"]
	final_predictions.to_csv("predictions/predictions_no_svd.csv", index=False) 





