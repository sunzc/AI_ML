#!/usr/bin/env python3
# Project: Building Naive Bayes Classifier for text
# Author: Zhichuang Sun
# Date: 2016.11.30

import os
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from nltk.stem.porter import *
import matplotlib.pyplot as plt

class Classifier:
	def __init__(self, categories, use_stemmer):
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
		self.test_data, self.test_target = self.get_data_target(categories, 'Test', use_stemmer)
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

	def get_feature_vector(self, bound, use_unigram, use_tfidf, use_stopwords):
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
			return X_tfidf
		else:
			return X
		
	def train_nb(self, percent, unigram_only, use_tfidf, use_stopwords):
		bound = int(len(self.train_data)/100) * percent

		X_u = self.get_feature_vector(bound, True, use_tfidf, use_stopwords)
		self.uclf = MultinomialNB().fit(X_u, self.train_target[:bound])

		if not unigram_only:
			X_b = self.get_feature_vector(bound, False, use_tfidf, use_stopwords)
			self.bclf = MultinomialNB().fit(X_b, self.train_target[:bound])

	def get_result(self):
		res = []
		u_res = []
		b_res = []
		X_test_counts_u = self.unigram_count_vect.transform(self.test_data)
		X_test_counts_b = self.bigram_count_vect.transform(self.test_data)
		upredicted = self.uclf.predict(X_test_counts_u)
		bpredicted = self.bclf.predict(X_test_counts_b)

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

	def train_svm(self, percent, unigram_only, use_tfidf, use_stopwords):
		bound = int(len(self.train_data)/100) * percent

		X_u = self.get_feature_vector(bound, True, use_tfidf, use_stopwords)
		self.uclf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42).fit(X_u, self.train_target[:bound])

		if not unigram_only:
			X_b = self.get_feature_vector(bound, False, use_tfidf, use_stopwords)
			self.bclf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42).fit(X_b, self.train_target[:bound])

	def train_lr(self, percent, unigram_only, use_tfidf, use_stopwords):
		bound = int(len(self.train_data)/100) * percent

		X_u = self.get_feature_vector(bound, True, use_tfidf, use_stopwords)
		self.uclf = LogisticRegression(C = 100, penalty='l2',tol = 0.01).fit(X_u, self.train_target[:bound])

		if not unigram_only:
			X_b = self.get_feature_vector(bound, False, use_tfidf, use_stopwords)
			self.bclf = LogisticRegression(C = 100, penalty='l2',tol = 0.01).fit(X_b, self.train_target[:bound])

	def train_rf(self, percent, unigram_only, use_tfidf, use_stopwords):
		bound = int(len(self.train_data)/100) * percent

		X_u = self.get_feature_vector(bound, True, use_tfidf, use_stopwords)
		self.uclf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0).fit(X_u, self.train_target[:bound])

		if not unigram_only:
			X_b = self.get_feature_vector(bound, False, use_tfidf, use_stopwords)
			self.bclf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0).fit(X_b, self.train_target[:bound])

	def get_performance_table(self, use_tfidf, use_stopwords):
		# for each classifier, two configurations: unigram and bigram
		# for each configuration, three parameters precision, recall, f1
		perf_table = []
		cla_names = []

		self.train_nb(100, False, use_tfidf, use_stopwords)
		res = self.get_result()
		perf_table.append(res)
		cla_names.append("Naive Bayes")

		#self.train_svm(100, False, use_tfidf, use_stopwords)
		#res = self.get_result()
		#perf_table.append(res)
		#cla_names.append("Support Vector Machine")

		#self.train_lr(100, False, use_tfidf, use_stopwords)
		#res = self.get_result()
		#perf_table.append(res)
		#cla_names.append("Logistic Regression")

		#self.train_rf(100, False, use_tfidf, use_stopwords)
		#res = self.get_result()
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

	def draw_learning_curve(self):
		plt.figure()
		plt.title("learning curve")
		plt.ylim((0.0, 1.1))
		plt.xlabel('Training Data Size')
		plt.ylabel('F1 Score')
		plt.grid()

		# size of training data
		x_axis = [x*10 for x in range(1,11)]
		# F1 score
		y_axis = []

		# NB
		nb_y_axis = []
		for x in x_axis:
			self.train_nb(x, True)
			X_test_counts_u = self.unigram_count_vect.transform(self.test_data)
			upredicted = self.uclf.predict(X_test_counts_u)
			nb_y_axis.append(metrics.f1_score(self.test_target, upredicted, average='macro'))
		plt.plot([x*len(self.train_data)/10 for x in range(1, 11)], nb_y_axis, 'o-', color="r", label="Naive Bayes")

		# LR
		lr_y_axis = []
		for x in x_axis:
			self.train_lr(x, True)
			X_test_counts_u = self.unigram_count_vect.transform(self.test_data)
			upredicted = self.uclf.predict(X_test_counts_u)
			lr_y_axis.append(metrics.f1_score(self.test_target, upredicted, average='macro'))
		plt.plot([x*len(self.train_data)/10 for x in range(1, 11)], lr_y_axis, 'o-', color="b", label="Logistical Regression")

		# svm
		svm_y_axis = []
		for x in x_axis:
			self.train_svm(x, True)
			X_test_counts_u = self.unigram_count_vect.transform(self.test_data)
			upredicted = self.uclf.predict(X_test_counts_u)
			svm_y_axis.append(metrics.f1_score(self.test_target, upredicted, average='macro'))
		plt.plot([x*len(self.train_data)/10 for x in range(1, 11)], svm_y_axis, 'o-', color="g", label="Support Vector Machine")

		# rf
		rf_y_axis = []
		for x in x_axis:
			self.train_rf(x, True)
			X_test_counts_u = self.unigram_count_vect.transform(self.test_data)
			upredicted = self.uclf.predict(X_test_counts_u)
			rf_y_axis.append(metrics.f1_score(self.test_target, upredicted, average='macro'))
		plt.plot([x*len(self.train_data)/10 for x in range(1, 11)], rf_y_axis, 'o-', color="y", label="Random Forest")
		plt.legend(loc="best")
		plt.show()

if __name__ == '__main__':
	categories = ['rec.sport.hockey','sci.med','soc.religion.christian','talk.religion.misc']
	cla_no_stemmer = Classifier(categories, False)

	print("Use Count Vector, No Stop Words, No Stemmer")
	cla_no_stemmer.get_performance_table(False, False)

	print("")

	print("Use Count Vector, Stop Words, No Stemmer")
	cla_no_stemmer.get_performance_table(False, True)

	print("")

	print("Use TFIDF Vector, No Stop Words, No Stemmer")
	cla_no_stemmer.get_performance_table(True, False)

	print("")

	print("Use TFIDF Vector, Stop Words, No Stemmer")
	cla_no_stemmer.get_performance_table(True, True)

	cla_with_stemmer = Classifier(categories, True)

	print("Use Count Vector, No Stop Words, Use Stemmer")
	cla_with_stemmer.get_performance_table(False, False)

	print("")

	print("Use Count Vector, Stop Words, Use Stemmer")
	cla_with_stemmer.get_performance_table(False, True)

	print("")

	print("Use TFIDF Vector, No Stop Words, Use Stemmer")
	cla_with_stemmer.get_performance_table(True, False)

	print("")

	print("Use TFIDF Vector, Stop Words, Use Stemmer")
	cla_with_stemmer.get_performance_table(True, True)
	#cla.draw_learning_curve()
