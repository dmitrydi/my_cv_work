# Класс для дообучения детектора на основе Darkflow
from darkflow.net.build import TFNet
import cv2
import skvideo.io
from sort.sort import *
import numpy as np
from .deep_sort.deep_sort.detection import Detection
#from .deep_sort.deep_sort.tracker import Tracker
from .deep_sort.generate_detections import *
#from .deep_sort import nn_matching

class DeepDetector(object):
	def __init__(self, options):
		self.detector = TFNet(options)

	def retrain(self, train_options):
		trainer_allower = {'train': True}
		self.detector.FLAGS.update(trainer_allower)
		self.detector.FLAGS.update(train_options)
		self.losses, self.train_losses = self.detector.train()

	def load_weights(self, load_options):
		trainer_stopper = {'train': False}
		self.detector.FLAGS.update(trainer_stopper)
		self.detector.FLAGS.update(load_options)

	def detect_and_save(self, tracker, track_encoder,
		encoder=None, tpe_predictor=None, save_mode='images', show_ids=True, verbose = True, video_file_path='', img_saving_path='',
		embs_saving_path=None, emb_dump_size=1000, max_imgs_per_person=50, delay=1, show_video=False):
		'''		
		детекция и сохранение либо изображений объектов, либо их низкоразмерных представлений (с помощью encoder)
		save_mode = ['images', 'embeddings', 'all']
		tracker = Tracker(metrics) - deep tracker
		metrics = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget) - местрика из модели deep_sort
		track_encoder = create_image_encoder(model_file_path, batch_size=32, loss='cosine') - кдировщих изображений для трекера (deep sort)
		encoder - кодировщик изображений перед tpe-кодировщиком
		tpe_predictor - возвращает tpe-фичи для модели верификации
		save_mode - что нужно сохранять, images, embeddings (tpe-фичи) или all
		show_video - [True, False] - показывать ли видео (чеез cv2)
		show_ids - показывать ли id объектов (при show_video=True)
		video_file_path - путь к папке с видеофайлами для обработки
		img_savimg_path - куда сохранять детектированные изображения (кропы)
		emb_saving_path - куда сохранять tpe-фичи если save_mode == 'embeddings', 'all'
		emb_dump_size - максимальное кол-во tpe-фич для изображений, когда происходит их сохранение в npy - архиве
		max_imgs_per_person - максимальное кол-во сохраняемых изображений для персоны
		delay - задержка при отбражении видео
		'''
		k = None
		embeddings = None
		emb_arch_counter = 0 #счетчик npy-архивов для сохранения низкоразмерных представлений
		assert save_mode in ['images', 'embeddings', 'all']
		for file in os.listdir(video_file_path):
			if verbose:
				print('processing file: {}'.format(file))
			#camera = skvideo.io.VideoCapture(os.path.join(video_file_path,file))
			frames = skvideo.io.FFmpegReader(os.path.join(video_file_path,file)).nextFrame()
			for img in frames:
				img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
				pred_dict = self.detector.return_predict(img)
				boxes = boxes_from_dict(pred_dict)
				if len(boxes) == 0:
					continue
				detections = gen_detections_from_image(track_encoder, img_bgr, boxes) 
				#bboxes = tracker.update(detections)
				tracker.predict()
				tracker.update(detections)

				for track in tracker.tracks:
					#print('strat processing boxes')
					box = track.to_tlwh()
					person_id = track.track_id
					person_dir = os.path.join(img_saving_path, str(person_id))
					
					if not os.path.exists(person_dir):
						os.mkdir(person_dir)
						#print('preson dir made for id {}'.format(person_id))
					

					n_img = len(os.listdir(person_dir))
					im_name = str(person_id) + '_' + str(n_img) + '.jpg'
					box = box.astype('int')
					imCrop = img_bgr[box[1]:(box[1]+box[3]), box[0]:(box[0]+box[2])]
					#imCrop = cv2.cvtColor(imCrop, cv2.COLOR_RGB2BGR)
					
					if n_img < max_imgs_per_person:
						if save_mode == 'images' or save_mode == 'all':
							cv2.imwrite(os.path.join(person_dir,im_name), imCrop)
						if save_mode == 'embeddings' or save_mode == 'all':
							embedding = encoder.encode(imCrop).astype(float)
							embedding = tpe_predictor.predict(embedding)
							embedding = np.concatenate((np.array([person_id], dtype=float)[np.newaxis],
										np.array(imCrop.shape)[np.newaxis], embedding), axis=1)
							if embeddings is None:
								embeddings = embedding#[np.newaxis]
							else:
								embeddings = np.vstack([embeddings, embedding])
							if embeddings.shape[0] >= emb_dump_size:
								npy_outfile = ('embs-'+str(emb_arch_counter*emb_dump_size)+
										'-'+str((emb_arch_counter+1)*emb_dump_size)+'.npy')
								np.save(os.path.join(embs_saving_path,npy_outfile), embeddings)
								embeddings = None
								emb_arch_counter += 1


				if show_video:
					cv2.rectangle(img_bgr, (box[0], box[1]), (box[2], box[3]), (255,0,0), 1)
					if show_ids:
						cv2.putText(img_bgr, 'id:' +str(box[4]),(box[0],box[1]),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0))
					cv2.imshow('ex', img_bgr)
					k = cv2.waitKey(delay)
					if k & 0xFF == ord('q'):
						camera.release()
						break

			if show_video:
				if k & 0xFF == ord('q'):
					break

		if show_video:
			cv2.destroyAllWindows()

def boxes_from_dict(pred_dict):
	out_dict = []
	for pred in pred_dict:
		if pred['label']=='person':
			x = int(pred['topleft']['x'])
			y = int(pred['topleft']['y'])
			w = int(pred['bottomright']['x'] - x)
			h = int(pred['bottomright']['y'] - y)
			conf = float(pred['confidence'])
			out_dict.append({'tlwh': (x,y,w,h), 'conf': conf})
	return out_dict

def gen_detections_from_image(img_encoder, img_bgr, boxes):
	cnn_shape = (128, 64)
	img_patches = []
	clean_boxes = []
	confs = []
	detections = []
	for box in boxes:
		box_ = box['tlwh']
		clean_boxes.append(box_)
		confs.append(box['conf'])
		patch = extract_image_patch(img_bgr, box_, cnn_shape)
		if not patch is None:
			img_patches.append(patch)
		else:
			img_patches.append(np.zeros_like(box_.shape.append(3)))
	img_patches = np.asarray(img_patches)
	features = img_encoder.encode(img_patches)
	for box, conf, feature in zip(clean_boxes, confs, features):
		detections.append(Detection(box, conf, feature))
	return detections

	

def get_boxes_from_dict(pred_dict):
	detections = None
	for pred in pred_dict:
		if pred['label'] == 'person':
			bbox = np.array(
				[int(pred['topleft']['x']),
				 pred['topleft']['y'],
				 pred['bottomright']['x'],
				 pred['bottomright']['y'],
				 pred['confidence']])
			if detections is None:
				detections = bbox[np.newaxis,:]
			else:
				detections = np.vstack([detections, bbox])
	if detections is None:
		detections = np.array([])
	return detections

