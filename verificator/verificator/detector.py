# Класс для дообучения детектора на основе Darkflow
from darkflow.net.build import TFNet
import cv2
import skvideo.io
from sort.sort import *
import numpy as np

class Detector(object):
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

	def detect_and_save(self, tracker, encoder=None, tpe_predictor=None, save_mode='images', show_ids=True, verbose = True, video_file_path='', img_saving_path='',
		embs_saving_path=None, emb_dump_size=1000, max_imgs_per_person=50, delay=1, show_video=False):
		# детекция и сохранение либо изображений объектов, либо их низкоразмерных представлений (с помощью encoder)
		# save_mode = ['images', 'embeddings', 'all']
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
				detections = get_boxes_from_dict(pred_dict)
				bboxes = tracker.update(detections)

				for box_ in bboxes:
					#print('strat processing boxes')
					box = box_.astype(int)
					person_id = box[4]
					person_dir = os.path.join(img_saving_path, str(person_id))
					
					if not os.path.exists(person_dir):
						os.mkdir(person_dir)
						#print('preson dir made for id {}'.format(person_id))
					

					n_img = len(os.listdir(person_dir))
					im_name = str(person_id) + '_' + str(n_img) + '.jpg'
					imCrop = img_bgr[box[1]:box[3], box[0]:box[2]]
					#imCrop = cv2.cvtColor(imCrop, cv2.COLOR_RGB2BGR)
					
					if n_img < max_imgs_per_person:
						if save_mode == 'images' or save_mode == 'all':
							cv2.imwrite(os.path.join(person_dir,im_name), imCrop)
						if save_mode == 'embeddings' or save_mode == 'all':
							embedding = encoder.encode(imCrop).astype(float)
							embedding = tpe_predictor.predict(embedding)
							embedding = np.concatenate((np.array([person_id], dtype=float)[np.newaxis], np.array(imCrop.shape)[np.newaxis], embedding), axis=1)
							if embeddings is None:
								embeddings = embedding#[np.newaxis]
							else:
								embeddings = np.vstack([embeddings, embedding])
							if embeddings.shape[0] >= emb_dump_size:
								npy_outfile = 'embs-'+str(emb_arch_counter*emb_dump_size)+'-'+str((emb_arch_counter+1)*emb_dump_size)+'.npy'
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

