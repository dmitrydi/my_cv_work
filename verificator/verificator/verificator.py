# Класс для повторного распознавания персон
#from metrics_utils import check_metrics
import cv2
import numpy

class Verificator(object):
	def __init__(self, encoder, predictor):
	# encoder - модель, преобразующая изображение в представление низкой размерности
	# predictor - модель, переводящая представление низкой размерности в пространство tpe - features
		self.encoder = encoder
		self.predictor = predictor

	def verify(self, img1, img2, distance):
	# возвращает true, если img1 и img2 принадлжеат одной персоне, False - иначе
	# img1, img2 - изображения двух персон
	# distance - граничная мера близости изображений
		img_ary1 = cv2.imread(img1)
		img_ary2 = cv2.imread(img2)
		features1 = self.encoder.encode(img_ary1)
		features2 = self.encoder.encode(img_ary2)
		tpe_features1 = self.predictor.predict(features1)
		tpe_features2 = self.predictor.predict(features2)
		scores = tpe_features1 @ tpe_features2.T
		return scores >= distance, scores

	def postprocess_data(self, archive_data_path, distance, min_w=None, min_h=None, min_ratio = 1.):
		'''
		постпроцессинг полученных архивов с embedding-ами:
		1) удаление малоразмерных изображений либо по ширине/высоте, либо по ratio
		2) перепроверка того, что одна и та же персона не попала в разные id из-за погрешностей трекера

		'''

		pass
