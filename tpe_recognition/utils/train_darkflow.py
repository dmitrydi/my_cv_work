import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
annotations_dir = '../data/cam_1_xml_hand'
images_dir = '../data/cam_1_imgs'

from darkflow.net.build import TFNet

build_path = 'test_build'
weights = os.path.join(build_path, 'bin', 'yolo.weights')
model = os.path.join(build_path, 'cfg', 'yolo.cfg')
cfg = os.path.join(build_path, 'cfg')
labels = os.path.join(build_path, 'labels.txt')
print(cfg)
options=dict()
options['model'] = model
options['load'] = weights
options['config'] = cfg
options['labels'] = labels
options['train'] = True
options['dataset'] = images_dir
options['annotation'] = annotations_dir
options['epoch'] = 1
options['gpu'] = 1
options["batch"] = 4
options["subdivisions"] = 1
tfnet = TFNet(options)

# вот здесь он и вылетает
tfnet.train()
