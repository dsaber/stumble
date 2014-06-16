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
	
	cv = TfidfVectorizer(stop_words="english", min_df=min_df, ngram_range=ngram_range, use_idf=True,
						 smooth_idf=True, sublinear_tf=True, strip_accents="unicode")
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

	print "loading in data"
	df = pd.read_csv("data/train_url.tsv", sep="\t")
	X, Y = make_xy(df)
	all_docs = text_extractor(X)
	X_test = pd.read_csv("data/test_url.tsv", sep="\t")
	all_test_docs = text_extractor(X_test)


	print "building TF-IDF's"
	cv_mat_title, test_title = tfidf_and_svd(all_docs["title"], all_test_docs["title"], 10, False, 1)
	cv_mat_url, test_url = tfidf_and_svd(all_docs["url"], all_test_docs["url"], 10, False, 3)
	cv_mat_body, test_body = tfidf_and_svd(all_docs["body"], all_test_docs["body"], 100, False, 9) 


	print "performing LSA"
	all_tf_train = hstack([cv_mat_title, cv_mat_url, cv_mat_body])
	all_tf_test = hstack([test_title, test_url, test_body])
	sd = TruncatedSVD(500, random_state=778)
	sd.fit(vstack([all_tf_train, all_tf_test]))
	training_features = sd.transform(all_tf_train)
	test_features = sd.transform(all_tf_test)


	print "fitting logistic regression"
	logreg = LogisticRegression(penalty="l2", dual=True, tol=0.0001)
	logreg.fit(training_features, Y)

	
	print "making predictions and outputting to CSV"
	final_predictions = pd.DataFrame(X_test["urlid"])
	final_predictions["label"] = logreg.predict_proba(test_features)[:, 1]
	final_predictions.to_csv("predictions/predictions_no_svd.csv", index=False) 





