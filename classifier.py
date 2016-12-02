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
import matplotlib.pyplot as plt

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

	def train_nb(self, percent, unigram_only):
		bound = int(len(self.train_data)/100) * percent

		self.unigram_count_vect = CountVectorizer()
		X_u = self.unigram_count_vect.fit_transform(self.train_data[:bound])
		self.uclf = MultinomialNB().fit(X_u, self.train_target[:bound])

		if not unigram_only:
			self.bigram_count_vect = CountVectorizer(ngram_range=(2, 2))
			X_b = self.bigram_count_vect.fit_transform(self.train_data[:bound])
			self.bclf = MultinomialNB().fit(X_b, self.train_target[:bound])

	def get_result(self):
		res = []
		u_res = []
		b_res = []
		X_test_counts_u = self.unigram_count_vect.transform(self.test_data)
		X_test_counts_b = self.bigram_count_vect.transform(self.test_data)
		upredicted = self.uclf.predict(X_test_counts_u)
		bpredicted = self.bclf.predict(X_test_counts_b)

		print(metrics.classification_report(self.test_target, upredicted, target_names=self.target_names))
		u_precision = metrics.precision_score(self.test_target, upredicted, average='macro')
		u_res.append(u_precision)
		u_recall = metrics.recall_score(self.test_target, upredicted, average='macro')
		u_res.append(u_recall)
		u_f1 = metrics.f1_score(self.test_target, upredicted, average='macro')
		u_res.append(u_f1)

		print(metrics.classification_report(self.test_target, bpredicted, target_names=self.target_names))
		b_precision = metrics.precision_score(self.test_target, bpredicted, average='macro')
		b_res.append(b_precision)
		b_recall = metrics.recall_score(self.test_target, bpredicted, average='macro')
		b_res.append(b_recall)
		b_f1 = metrics.f1_score(self.test_target, bpredicted, average='macro')
		b_res.append(b_f1)

		res.append(u_res)
		res.append(b_res)

		return res

	def train_svm(self, percent, unigram_only):
		bound = int(len(self.train_data)/100) * percent

		self.unigram_count_vect = CountVectorizer()
		X_u = self.unigram_count_vect.fit_transform(self.train_data[:bound])
		self.uclf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42).fit(X_u, self.train_target[:bound])

		if not unigram_only:
			self.bigram_count_vect = CountVectorizer(ngram_range=(2, 2))
			X_b = self.bigram_count_vect.fit_transform(self.train_data[:bound])
			self.bclf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42).fit(X_b, self.train_target[:bound])

	def train_lr(self, percent, unigram_only):
		bound = int(len(self.train_data)/100) * percent

		self.unigram_count_vect = CountVectorizer()
		X_u = self.unigram_count_vect.fit_transform(self.train_data[:bound])
		self.uclf = LogisticRegression(C = 100, penalty='l2',tol = 0.01).fit(X_u, self.train_target[:bound])

		if not unigram_only:
			self.bigram_count_vect = CountVectorizer(ngram_range=(2, 2))
			X_b = self.bigram_count_vect.fit_transform(self.train_data[:bound])
			self.bclf = LogisticRegression(C = 100, penalty='l2',tol = 0.01).fit(X_b, self.train_target[:bound])

	def train_rf(self, percent, unigram_only):
		bound = int(len(self.train_data)/100) * percent

		self.unigram_count_vect = CountVectorizer()
		X_u = self.unigram_count_vect.fit_transform(self.train_data[:bound])
		self.uclf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0).fit(X_u, self.train_target[:bound])

		if not unigram_only:
			self.bigram_count_vect = CountVectorizer(ngram_range=(2, 2))
			X_b = self.bigram_count_vect.fit_transform(self.train_data[:bound])
			self.bclf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0).fit(X_b, self.train_target[:bound])

	def get_performance_table(self):
		# for each classifier, two configurations: unigram and bigram
		# for each configuration, three parameters precision, recall, f1
		perf_table = []
		cla_names = []

		self.train_nb(100, False)
		res = self.get_result()
		perf_table.append(res)
		cla_names.append("Naive Bayes")

		self.train_svm(100, False)
		res = self.get_result()
		perf_table.append(res)
		cla_names.append("Support Vector Machine")

		self.train_lr(100, False)
		res = self.get_result()
		perf_table.append(res)
		cla_names.append("Logistic Regression")

		self.train_rf(100, False)
		res = self.get_result()
		perf_table.append(res)
		cla_names.append("Random Forest")

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
	cla = Classifier(categories)
	cla.get_performance_table()
	#cla.draw_learning_curve()
	#cla.train_nb(100)
	#cla.get_result()
	#cla.train_svm(100)
	#cla.get_result()
	#cla.train_lr(100)
	#cla.get_result()
	#cla.train_rf(100)
	#cla.get_result()
