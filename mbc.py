#!/usr/bin/env python3
# Project: Building Naive Bayes Classifier for text
# Author: Zhichuang Sun
# Date: 2016.11.30

import os
import sys
import random
import pickle
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from nltk.stem.porter import *
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt

class Classifier:
	def __init__(self, categories, use_stemmer, subset):
		"""
		train data and test data
		data directory hiearachy:
		data/
			Training/
				rec.sport.hockey/
				... (other categorities)
			Test/
				rec.sport.hockey/
				... (other categorities)
		preprocess data into feature set that can be used by NB classifiers
		we need to support unigram baseline and bigram baseline	
		"""
		self.train_data, self.train_target = self.get_data_target(categories, 'Training', use_stemmer)
		self.test_data, self.test_target = self.get_data_target(categories, subset, use_stemmer)
		self.target_names = categories

	def get_data_target(self, categories, subset, use_stemmer):
		data = []
		target = []
		idx = 0
		data_home = os.path.join(os.getcwd(),'data')
		for category in categories:
			idx = categories.index(category)
			cat_path = os.path.join(os.getcwd(),data_home,subset,category)
			filenames = os.listdir(cat_path)
			for filename in filenames:
				#print(filename)
				f = open(os.path.join(cat_path, filename), 'rb')
				content = self.escape_header(f.read().decode('utf-8', 'ignore'), use_stemmer)
				#content = '\n'.join((f.read().encode('utf-8')).split('\n'[5:]))
				data.append(content)
				target.append(idx)
		shuffled_idx = list(range(len(data)))
		random.shuffle(shuffled_idx)
		shuffled_data = []
		shuffled_target = []
		for i in shuffled_idx:
			shuffled_data.append(data[i])
			shuffled_target.append(target[i])
		return shuffled_data, shuffled_target
			
	def escape_header(self, file_content, use_stemmer):
		# escape the header (first 5 lines)
		#
		escaped_con = '\n'.join(file_content.split('\n')[5:])

		if use_stemmer:
			stemmer = PorterStemmer()
			singles = [stemmer.stem(word) for word in escaped_con.split()]
			stemmed_con = ' '.join(singles)
			#print(stemmed_con)
			return stemmed_con
		else:
			return escaped_con

	def get_feature_vector(self, bound, use_unigram, use_tfidf, use_stopwords, select_feature):
		res = []
		stopwords = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]

		if use_unigram:
			if use_stopwords:
				self.unigram_count_vect = CountVectorizer(stop_words = stopwords)
			else:
				self.unigram_count_vect = CountVectorizer()
			X = self.unigram_count_vect.fit_transform(self.train_data[:bound])
		else:
			if use_stopwords:
				self.bigram_count_vect = CountVectorizer(stop_words = stopwords, ngram_range=(2, 2))
			else:
				self.bigram_count_vect = CountVectorizer(ngram_range=(2, 2))
			X = self.bigram_count_vect.fit_transform(self.train_data[:bound])
		
		if use_tfidf:
			tfidf_transformer = TfidfTransformer()
			X_tfidf = tfidf_transformer.fit_transform(X)
			res = X_tfidf
		else:
			res = X

		# print("before selection : "+ str(res.shape))
		if select_feature:
			# Doesn't work, non-boolean feature, can't meet thre·
			# sel = VarianceThreshold(threshold=(.4 * (1 - .4)))
			# res = sel.fit_transform(res)
			if use_unigram:
				self.u_ch2 = SelectKBest(chi2, k=4000)
				res = self.u_ch2.fit_transform(res, self.train_target[:bound])
			else:
				self.b_ch2 = SelectKBest(chi2, k=4000)
				res = self.b_ch2.fit_transform(res, self.train_target[:bound])
			#print("after selection : "+ str(res.shape))

		return res

		
	def train_nb(self, percent, unigram_only, use_tfidf, use_stopwords, select_feature):
		bound = int(len(self.train_data)/100) * percent

		X_u = self.get_feature_vector(bound, True, use_tfidf, use_stopwords, select_feature)
		self.uclf = MultinomialNB().fit(X_u, self.train_target[:bound])

		if not unigram_only:
			X_b = self.get_feature_vector(bound, False, use_tfidf, use_stopwords, select_feature)
			self.bclf = MultinomialNB().fit(X_b, self.train_target[:bound])

	def get_result(self, select_feature):
		res = []
		u_res = []
		b_res = []
		X_u = self.unigram_count_vect.transform(self.test_data)
		X_b = self.bigram_count_vect.transform(self.test_data)
		if select_feature:
			X_u = self.u_ch2.transform(X_u)
			X_b = self.b_ch2.transform(X_b)
		upredicted = self.uclf.predict(X_u)
		bpredicted = self.bclf.predict(X_b)

		#print(metrics.classification_report(self.test_target, upredicted, target_names=self.target_names))
		u_precision = metrics.precision_score(self.test_target, upredicted, average='macro')
		u_res.append(u_precision)
		u_recall = metrics.recall_score(self.test_target, upredicted, average='macro')
		u_res.append(u_recall)
		u_f1 = metrics.f1_score(self.test_target, upredicted, average='macro')
		u_res.append(u_f1)

		#print(metrics.classification_report(self.test_target, bpredicted, target_names=self.target_names))
		b_precision = metrics.precision_score(self.test_target, bpredicted, average='macro')
		b_res.append(b_precision)
		b_recall = metrics.recall_score(self.test_target, bpredicted, average='macro')
		b_res.append(b_recall)
		b_f1 = metrics.f1_score(self.test_target, bpredicted, average='macro')
		b_res.append(b_f1)

		res.append(u_res)
		res.append(b_res)

		return res

	def train_svm(self, percent, unigram_only, use_tfidf, use_stopwords, select_feature):
		bound = int(len(self.train_data)/100) * percent

		X_u = self.get_feature_vector(bound, True, use_tfidf, use_stopwords, select_feature)
		self.uclf = SVC(kernel='linear', class_weight='balanced')
		#self.uclf = SVC(kernel='rbf')
		self.uclf.fit(X_u, self.train_target[:bound])

		#t0 = time()
		#param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
		#self.uclf = GridSearchCV(SVC(kernel='linear', class_weight='balanced'), param_grid)
		#self.uclf = self.uclf.fit(X_u, self.train_target[:bound])
		#print("u done in %0.3fs" % (time() - t0))

		if not unigram_only:
			#t0 = time()
			X_b = self.get_feature_vector(bound, False, use_tfidf, use_stopwords, select_feature)
			#self.bclf = SVC(kernel='rbf')
			self.bclf = SVC(kernel='linear', class_weight='balanced')
			self.bclf.fit(X_b, self.train_target[:bound])
			#self.bclf = GridSearchCV(SVC(kernel='linear', class_weight='balanced'), param_grid)
			#self.bclf = self.uclf.fit(X_b, self.train_target[:bound])
			#print("b done in %0.3fs" % (time() - t0))

	def train_lr(self, percent, unigram_only, use_tfidf, use_stopwords, select_feature):
		bound = int(len(self.train_data)/100) * percent

		X_u = self.get_feature_vector(bound, True, use_tfidf, use_stopwords, select_feature)
		self.uclf = LogisticRegression(C = 100, penalty='l2',tol = 0.01).fit(X_u, self.train_target[:bound])

		if not unigram_only:
			X_b = self.get_feature_vector(bound, False, use_tfidf, use_stopwords, select_feature)
			self.bclf = LogisticRegression(C = 100, penalty='l2',tol = 0.01).fit(X_b, self.train_target[:bound])

	def train_rf(self, percent, unigram_only, use_tfidf, use_stopwords, select_feature):
		bound = int(len(self.train_data)/100) * percent

		X_u = self.get_feature_vector(bound, True, use_tfidf, use_stopwords, select_feature)
		self.uclf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0).fit(X_u, self.train_target[:bound])

		if not unigram_only:
			X_b = self.get_feature_vector(bound, False, use_tfidf, use_stopwords, select_feature)
			self.bclf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0).fit(X_b, self.train_target[:bound])

	def get_performance_table(self, use_tfidf, use_stopwords, select_feature):
		# for each classifier, two configurations: unigram and bigram
		# for each configuration, three parameters precision, recall, f1
		perf_table = []
		cla_names = []

		self.train_nb(100, False, use_tfidf, use_stopwords, select_feature)
		res = self.get_result(select_feature)
		perf_table.append(res)
		cla_names.append("Naive Bayes")

		#self.train_svm(100, False, use_tfidf, use_stopwords, select_feature)
		#res = self.get_result(select_feature)
		#perf_table.append(res)
		#cla_names.append("Support Vector Machine")

		#self.train_lr(100, False, use_tfidf, use_stopwords, select_feature)
		#res = self.get_result(select_feature)
		#perf_table.append(res)
		#cla_names.append("Logistic Regression")

		#self.train_rf(100, False, use_tfidf, use_stopwords, select_feature)
		#res = self.get_result(select_feature)
		#perf_table.append(res)
		#cla_names.append("Random Forest")

		i = 0
		conf_names = ["Unigram Baselines", "Bigram Baselines"]
		perf_names = ["Precision Score", "   Recall Score", "       F1 Score"]
		for res in perf_table:
			print(cla_names[i])
			j = 0
			for conf in res:
				print("    " + conf_names[j])
				k = 0
				for score in conf:
					print("    " + "    " + perf_names[k] + ":" + str(score))
					k += 1
				j += 1
			i += 1

	def do_predict(self, subset):
		perf_table = []
		cla_names = []

		self.test_data, self.test_target = self.get_data_target(self.target_names, subset, True)

		res = self.get_result(False)
		perf_table.append(res)
		cla_names.append("Naive Bayes")

		i = 0
		conf_names = ["Unigram Baselines", "Bigram Baselines"]
		perf_names = ["Precision Score", "   Recall Score", "       F1 Score"]
		for res in perf_table:
			print(cla_names[i])
			j = 0
			for conf in res:
				print("    " + conf_names[j])
				k = 0
				for score in conf:
					print("    " + "    " + perf_names[k] + ":" + str(score))
					k += 1
				j += 1
			i += 1


if __name__ == '__main__':
	categories = ['rec.sport.hockey','sci.med','soc.religion.christian','talk.religion.misc']
	if len(sys.argv) == 2:
		subset = sys.argv[1]
	else:
		print("Usage:")
		print("\t./mbc.py subset_name")
		exit(1)

	try:
		best_cla = pickle.load(open( "best_cla.p", "rb" ))
		best_cla.do_predict(subset)
	except:
		best_cla = Classifier(categories, use_stemmer=True, subset=subset)
		best_cla.get_performance_table(use_tfidf=False, use_stopwords=True, select_feature=False)
		pickle.dump(best_cla, open( "best_cla.p", "wb" ))

	#stemmer_array = [True, False]
	#stopwords_array = [True, False]
	#tfidf_array = [True, False]
	#select_feature_array = [True, False]

	#for has_stemmer in stemmer_array:
	#	cla = Classifier(categories, has_stemmer)
	#	for use_tfidf in tfidf_array:
	#		for use_stopwords in stopwords_array:
	#			for select_feature in select_feature_array:
	#				if has_stemmer == use_stopwords:
	#					print("Stemmer: " + str(has_stemmer), end=";")
	#					print("TFIDF: " + str(use_tfidf), end=";")
	#					print("Stop Words: " + str(use_stopwords), end=";")
	#					print("Select Feature: " + str(select_feature), end=";")
	#					cla.get_performance_table(use_tfidf, use_stopwords, select_feature)
	#					print("")
	#				else:
	#					continue
	#	cla = Classifier(categories, has_stemmer)
