import numpy as np
import random
import copy
class CoTrainingClassifier(object):
	"""
	Parameters:
	clf - The classifier that will be used in the cotraining algorithm on the X1 feature set
		(Note a copy of clf will be used on the X2 feature set if clf2 is not specified).

	clf2 - (Optional) A different classifier type can be specified to be used on the X2 feature set
		 if desired.

	p - (Optional) The number of positive examples that will be 'labeled' by each classifier during each iteration
		The default is the is determined by the smallest integer ratio of positive to negative samples in L (from paper)

	n - (Optional) The number of negative examples that will be 'labeled' by each classifier during each iteration
		The default is the is determined by the smallest integer ratio of positive to negative samples in L (from paper)

	k - (Optional) The number of iterations
		The default is 30 (from paper)

	u - (Optional) The size of the pool of unlabeled samples from which the classifier can choose
		Default - 75 (from paper)
	"""

	def __init__(self, clf, clf2=None, n=200, k=20, u=400):
		self.clf1_ = clf
		
		if clf2 == None:
			self.clf2_ = copy.copy(clf)
		else:
			self.clf2_ = clf2

		self.n_ = n
		self.k_ = k
		self.u_ = u

		random.seed()


	def fit(self, X1, X2, y):
		"""
		Description:
		fits the classifiers on the partially labeled data, y.

		Parameters:
		X1 - array-like (n_samples, n_features_1): first set of features for samples
		X2 - array-like (n_samples, n_features_2): second set of features for samples
		y - array-like (n_samples): labels for samples, -1 indicates unlabeled

		"""

		#we need y to be a numpy array so we can do more complex slicing
		y = np.asarray(y)

		assert(self.n_ > 0 and self.k_ > 0 and self.u_ > 0)

		#the set of unlabeled samples
		U = [i for i, y_i in enumerate(y) if y_i == -1]

		#we randomize here, and then just take from the back so we don't have to sample every time我们在这里随机化，然后从后面取样，这样我们就不必每次都取样
		np.random.shuffle(U)

		#this is U' in paper
		U_ = U[-min(len(U), self.u_):]

		#the samples that are initially labeled
		L = [i for i, y_i in enumerate(y) if y_i != -1]

		#remove the samples in U_ from U
		U = U[:-len(U_)]


		it = 0 #number of cotraining iterations we've done so far

		#loop until we have assigned labels to everything in U or we hit our iteration break condition循环，直到我们为U中的所有内容指定了标签，或者我们达到了迭代中断条件
		L1 = L
		L2 = L
		while it != self.k_ and U:
			it += 1

			self.clf1_.fit(X1[L1], y[L1])
			self.clf2_.fit(X2[L2], y[L2])

			y1_prob = self.clf1_.predict_proba(X1[U_])
			y2_prob = self.clf2_.predict_proba(X2[U_])


			n1, n2, p1, p2 = [], [], [], []

			for i in (y1_prob[:, 0].argsort())[-self.n_:]:  # y1预测为0的概率从小到大进行排序，选取后n个
				if y1_prob[i, 0] > 0.5:
					n1.append(i)  # 对n扩充，加上i
			for i in (y1_prob[:, 1].argsort())[-self.n_:]:
				if y1_prob[i, 1] > 0.5:
					p1.append(i)

			for i in (y2_prob[:, 0].argsort())[-self.n_:]:
				if y2_prob[i, 0] > 0.5:
					n2.append(i)
			for i in (y2_prob[:, 1].argsort())[-self.n_:]:
				if y2_prob[i, 1] > 0.5:
					p2.append(i)

			# label the samples and remove thes newly added samples from U_
			y[[U_[x] for x in p1]] = 1
			y[[U_[x] for x in p2]] = 1
			y[[U_[x] for x in n1]] = 0
			y[[U_[x] for x in n2]] = 0

			L1.extend([U_[x] for x in p2])
			L1.extend([U_[x] for x in n2])
			L2.extend([U_[x] for x in p1])
			L2.extend([U_[x] for x in n1])

			U_ = [elem for elem in U_ if not (elem in p1 or elem in n1 or elem in p2 or elem in n2)]

			# add new elements to U_
			add_counter = 0  # number we have added from U to U_
			num_to_add = len(p1) + len(n1) + len(p2) + len(n2)
			while add_counter != num_to_add and U:
				add_counter += 1
				U_.append(U.pop())


		#let's fit our final model
		self.clf1_.fit(X1[L1], y[L1])
		self.clf2_.fit(X2[L2], y[L2])


	def supports_proba(self, clf, x):
		"""Checks if a given classifier supports the 'predict_proba' method, given a single vector x"""
		try:
			clf.predict_proba([x])
			return True
		except:
			return False
	
	def predict(self, X1, X2):

		y1 = self.clf1_.predict(X1)
		y2 = self.clf2_.predict(X2)

		proba_supported = self.supports_proba(self.clf1_, X1[0]) and self.supports_proba(self.clf2_, X2[0])

		#fill y_pred with -1 so we can identify the samples in which the classifiers failed to agree用-1填充y_pred，这样我们就可以识别分类器不一致的样本
		y_pred = np.asarray([-1] * X1.shape[0])

		for i, (y1_i, y2_i) in enumerate(zip(y1, y2)):
			if y1_i == y2_i:
				y_pred[i] = y1_i
			elif proba_supported:
				y1_probs = self.clf1_.predict_proba([X1[i]])[0]
				y2_probs = self.clf2_.predict_proba([X2[i]])[0]
				sum_y_probs = [prob1 + prob2 for (prob1, prob2) in zip(y1_probs, y2_probs)]
				max_sum_prob = max(sum_y_probs)
				y_pred[i] = sum_y_probs.index(max_sum_prob)

			else:
				#the classifiers disagree and don't support probability, so we guess
				y_pred[i] = random.randint(0, 1)

			
		#check that we did everything right
		assert not (-1 in y_pred)

		return y_pred


	def predict_proba(self, X1, X2):
		"""Predict the probability of the samples belonging to each class."""
		y_proba = np.full((X1.shape[0], 2), -1)

		y1_proba = self.clf1_.predict_proba(X1)
		y2_proba = self.clf2_.predict_proba(X2)

		for i, (y1_i_dist, y2_i_dist) in enumerate(zip(y1_proba, y2_proba)):
			y_proba[i][0] = (y1_i_dist[0] + y2_i_dist[0]) / 2
			y_proba[i][1] = (y1_i_dist[1] + y2_i_dist[1]) / 2

		_epsilon = 0.0001
		assert all(abs(sum(y_dist) - 1) <= _epsilon for y_dist in y_proba)
		return y_proba
