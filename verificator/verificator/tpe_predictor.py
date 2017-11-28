# Класс для обучения классификатора на триплетах
from tpe_model.tpe_model_builder import build_tpe
from sklearn.decomposition import PCA
from tpe_model.tpe_train import *
import numpy as np


class TPEPredictor(object):
# класс для преобразования низкоразмерного представления изображения в tpe-пространство

	def __init__(self, n_in, n_out):
	# n_in - размерность входного представления
	# n_out - размерность выходного представления для tpe - модели
	# weights_path - путь к весам предобученной модели
		model, predictor = build_tpe(n_in, n_out)
		self.model = model
		self.predictor = predictor
		self.in_shape = n_in
		self.out_shape = n_out
		self.weights_loaded = False

	def load_weights(self, weights_path):
		self.model.load_weights(weights_path)
		self.weights_loaded = True

	def get_weights(self):
		return self.model.get_weights()

	def predict(self, img_embedding):
	# преобразование низкоразмерного представления в tpe-представление
		return self.predictor.predict(img_embedding)

	def train(
		self,
		nepoch,
		batch_size,
		sampling_batch_size=100,
		save_weights=True,
		weights_saving_path='',
		weights_saving_name='weights.h5',
		labels = None,
		embeddings = None,
		npy_embeddings = None,
		weights = None,
		train_test_split = 0.8
		):
		if labels is None and embeddings is None:
			self.labels, self.embs, self.nclasses, self.nfeatures, self.nexamples = unpack_features(npy_embeddings)
		elif npy_embeddings is None:
			self.labels = labels
			self.embs = embeddings
			self.nclasses = np.unique(self.labels).shape[0]
			self.nfeatures = self.embs[0].shape[0]
		else:
			raise ValueError('Either labels, embeddings or npy_embeddings should be provided')
		self.train_labels, self.train_embs, self.dev_labels, self.dev_embs = train_dev_split(self.embs, self.labels, train_test_split)
		if not self.weights_loaded:
			if weights is None:
				pca = PCA(self.out_shape)
				pca.fit(self.train_embs)
				W_pca = pca.components_
				self.model.set_weights([W_pca])
			else:
				self.model.load_weights(weights)
		# добавить опцию по созданию протокола
		self.dev_protocol = make_protocol(self.dev_labels)
		self.mineer, self.deer = train_tpe(self.model, self.predictor, self.train_embs, self.dev_embs,
    			self.dev_protocol, self.train_labels, self.nclasses, nepoch, batch_size, sampling_batch_size,
    			save_weights=save_weights, model_saving_path=weights_saving_path, model_saving_name = weights_saving_name)

