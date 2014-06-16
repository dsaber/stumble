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

import json

if __name__ == "__main__":

		df = pd.read_csv("data/train.tsv", sep="\t")
		df_test = pd.read_csv("data/test.tsv", sep="\t")
		urls = [] 

		for i in range(df.shape[0]):

			json_obj = json.loads(df["boilerplate"][i])
			if "url" in json_obj.keys():
				urls.append(json_obj["url"].split(" ")[0])

		urls = pd.Series(urls)
		df["url"] = urls 

		popular_urls = urls.value_counts() 
		
		for i, pop_url in enumerate(popular_urls.index.values):
			if i > 31:
				break

			df[pop_url] = np.zeros((df.shape[0], 1))
			df[pop_url][df["url"] == pop_url] = 1

			df_test[pop_url] = np.zeros((df_test.shape[0], 1))
			df_test[pop_url][df["url"] == pop_url] = 1

		print df.columns


		df.to_csv("data/train_url.tsv", sep="\t")
		df_test.to_csv("data/test_url.tsv", sep="\t")




			