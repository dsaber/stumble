import pandas as pd 
import numpy as np 

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
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


def text_extractor(X):

	return list(X["boilerplate"])


def specific_extraction(X, section):

	spec_docs = [] 

	for i in range(X.shape[0]):
		str_array = [] 
		json_obj = json.loads(X["boilerplate"][i])

		if section in json_obj.keys():
			spec_docs.append(json.dumps(json_obj[section]))
		else:
			spec_docs.append("x")

	return spec_docs


def tfidf_and_svd(train_docs, test_docs, num, svd=True, with_options=True):
	
	all_docs = train_docs + test_docs

	tf = TfidfVectorizer(stop_words="english", min_df=9, ngram_range=(1, 2), use_idf=with_options,
						 smooth_idf=with_options, sublinear_tf=with_options, strip_accents="unicode")
	tf.fit(all_docs)
	tf_mat_train = tf.transform(train_docs)
	tf_mat_test = tf.transform(test_docs)

	if svd is False:
		return tf_mat_train, tf_mat_test

	svd = TruncatedSVD(n_components=num)

	doc_result = svd.fit_transform(tf_mat_train)
	test_doc_result = svd.transform(tf_mat_test)

	return doc_result, test_doc_result



# Build Logistic Regression Model

if __name__ == "__main__":

	print "loading data in"
	df = pd.read_table("data/train.tsv", sep="\t")
	X_test = pd.read_table("data/test.tsv", sep="\t")
	X, Y = make_xy(df)

	print "extracting text"
	all_train_docs = text_extractor(df)
	all_test_docs = text_extractor(X_test)

	print "making TF-IDF and performing LSA"
	svd_train, svd_test = tfidf_and_svd(all_train_docs, all_test_docs, 500)

	print "running logistic regression"
	logreg = LogisticRegression(penalty="l2", dual=True, tol=0.0001)
	logreg.fit(svd_train, Y)

	print "making predictions and outputting to CSV"
	final_predictions = pd.DataFrame(X_test["urlid"])
	final_predictions["label"] = logreg.predict_proba(svd_test)[:, 1]
	final_predictions.to_csv("predictions/pred.csv", index=False)


	







