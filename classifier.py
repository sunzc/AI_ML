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
from sklearn import metrics

class Classifier:
	def __init__(self, categories):
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
		self.train_data, self.train_target = self.get_data_target(categories, 'Training')
		self.test_data, self.test_target = self.get_data_target(categories, 'Test')
		self.target_names = categories

	def get_data_target(self, categories, subset):
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
				content = self.escape_header(f.read().decode('utf-8', 'ignore'))
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
			
	def escape_header(self, file_content):
		# escape the header (first 5 lines)
		return '\n'.join(file_content.split('\n')[5:])

	def train_nb(self, percent):
		self.unigram_count_vect = CountVectorizer()
		self.bigram_count_vect = CountVectorizer(ngram_range=(2, 2))
		bound = int(len(self.train_data)/100) * percent
		X_u = self.unigram_count_vect.fit_transform(self.train_data[:bound])
		X_b = self.bigram_count_vect.fit_transform(self.train_data[:bound])
		self.uclf = MultinomialNB().fit(X_u, self.train_target[:bound])
		self.bclf = MultinomialNB().fit(X_b, self.train_target[:bound])

	def get_result(self):
		X_test_counts_u = self.unigram_count_vect.transform(self.test_data)
		X_test_counts_b = self.bigram_count_vect.transform(self.test_data)
		upredicted = self.uclf.predict(X_test_counts_u)
		bpredicted = self.bclf.predict(X_test_counts_b)
		print(metrics.classification_report(self.test_target, upredicted, target_names=self.target_names))
		print(metrics.precision_score(self.test_target, upredicted, average='macro'))
		print(metrics.recall_score(self.test_target, upredicted, average='macro'))
		print(metrics.f1_score(self.test_target, upredicted, average='macro'))

		print(metrics.classification_report(self.test_target, bpredicted, target_names=self.target_names))
		print(metrics.precision_score(self.test_target, bpredicted, average='macro'))
		print(metrics.recall_score(self.test_target, upredicted, average='macro'))
		print(metrics.f1_score(self.test_target, bpredicted, average='macro'))

	def train_svm(self, percent):
		self.unigram_count_vect = CountVectorizer()
		self.bigram_count_vect = CountVectorizer(ngram_range=(2, 2))
		bound = int(len(self.train_data)/100) * percent
		X_u = self.unigram_count_vect.fit_transform(self.train_data[:bound])
		X_b = self.bigram_count_vect.fit_transform(self.train_data[:bound])
		#X_train_counts_u = self.unigram_count_vect.fit_transform(self.train_data)
		#X_train_counts_b = self.bigram_count_vect.fit_transform(self.train_data)
		self.uclf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42).fit(X_u, self.train_target[:bound])
		self.bclf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42).fit(X_b, self.train_target[:bound])

	def train_lr(self, percent):
		self.unigram_count_vect = CountVectorizer()
		self.bigram_count_vect = CountVectorizer(ngram_range=(2, 2))
		bound = int(len(self.train_data)/100) * percent
		X_u = self.unigram_count_vect.fit_transform(self.train_data[:bound])
		X_b = self.bigram_count_vect.fit_transform(self.train_data[:bound])
		#X_u = self.unigram_count_vect.fit_transform(self.train_data)
		#X_b = self.bigram_count_vect.fit_transform(self.train_data)
		self.uclf = LogisticRegression(C = 100, penalty='l2',tol = 0.01).fit(X_u, self.train_target[:bound])
		self.bclf = LogisticRegression(C = 100, penalty='l2',tol = 0.01).fit(X_b, self.train_target[:bound])

	def train_rf(self, percent):
		self.unigram_count_vect = CountVectorizer()
		self.bigram_count_vect = CountVectorizer(ngram_range=(2, 2))
		bound = int(len(self.train_data)/100) * percent
		X_u = self.unigram_count_vect.fit_transform(self.train_data[:bound])
		X_b = self.bigram_count_vect.fit_transform(self.train_data[:bound])
		#X_u = self.unigram_count_vect.fit_transform(self.train_data)
		#X_b = self.bigram_count_vect.fit_transform(self.train_data)
		self.uclf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0).fit(X_u, self.train_target[:bound])
		self.bclf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0).fit(X_b, self.train_target[:bound])

if __name__ == '__main__':
	categories = ['rec.sport.hockey','sci.med','soc.religion.christian','talk.religion.misc']
	cla = Classifier(categories)
	cla.train_nb(100)
	cla.get_result()
	cla.train_svm(100)
	cla.get_result()
	cla.train_lr(100)
	cla.get_result()
	cla.train_rf(100)
	cla.get_result()
