from .deep_sort.generate_detections import *
import numpy as np

class DeepSortExtractor(object):
# класс для получения низкоразмерных свойств изображений в модели Deep Sort

	def __init__(self, model_path, batch_size=32, loss_mode='cosine'):
		self.model_path = model_path
		self.batch_size = batch_size
		self.loss_mode = loss_mode
		self.encoder = create_box_encoder(
			self.model_path, batch_size=self.batch_size, loss_mode=self.loss_mode)

	def encode(self, img_ary):
	# преобразование изображение в низкоразмерное представление
	# img_ary - изображение в виде numpy массива, channels last
	# возвращает 128-мерное представление изображения
		bbox = np.array([0,0,img_ary.shape[1], img_ary.shape[0]]).reshape(1,4)
		return self.encoder(img_ary, bbox)
