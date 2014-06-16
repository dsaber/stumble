import pandas as pd 
import numpy as np 

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression
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

	df = pd.read_table("data/train.tsv", sep="\t")
	df_test = pd.read_table("data/test.tsv", sep="\t")
	X, Y = make_xy(df)
	all_train_docs = list(np.array(df)[:, 2])
	all_test_docs = list(np.array(df_test)[:, 2])
	all_docs = all_train_docs + all_test_docs 	


	# make a bag of words
	tf = TfidfVectorizer(stop_words="english", min_df=3, ngram_range=(1, 2), use_idf=True,
						 smooth_idf=True, sublinear_tf=True, strip_accents="unicode")
	tf.fit(all_docs) 
	# make matrices for training and test sets
	tf_train = tf.transform(all_train_docs)
	tf_test = tf.transform(all_test_docs)

	svd = TruncatedSVD(n_components=527)
	svd_train = svd.fit_transform(tf_train)
	svd_test = svd.transform(tf_test)


	logreg = LogisticRegression(penalty="l2", dual=True, tol=0.0001)
	logreg.fit(svd_train, Y)


	final_predictions = pd.DataFrame(df_test["urlid"])
	final_predictions["label"] = logreg.predict_proba(svd_test)[:, 1]
	final_predictions.to_csv("predictions/basic_pred2.csv", index=False)


	







