import json, os, ntpath
from dict2xml import dict2xml
import numpy as np

def all_boxes_nonzero(bbox_list):
    ans = True
    if len(bbox_list) == 0:
        ans = False
    for bbox in bbox_list:
        ans = ans and (bbox['bottomright'][0]-bbox['topleft'][0])*(bbox['bottomright'][1]-bbox['topleft'][1])>0
    return ans

def some_boxes_nearbound(bbox_list, scale, img_shape):
    ans = False
    for bbox in bbox_list:
        ans = ans or (not check_scaled_box(bbox, scale, img_shape))
    return ans

def check_scaled_box(bbox, scale, img_shape):
    x_max, y_max, _ = img_shape
    x_min, y_min = 0., 0.
    shift_x = (scale-1.)*x_max/2.
    shift_y = (scale - 1.)*y_max/2.
    x_b_min, y_b_min = bbox['topleft']
    x_b_max, y_b_max = bbox['bottomright']
    x_b_min, x_b_max = scale*x_b_min - shift_x, scale*x_b_max - shift_x
    y_b_min, y_b_max = scale*y_b_min - shift_y, scale*y_b_max - shift_y
    return (x_b_min*y_b_min > 0. and (x_b_max-x_max)*(y_b_max-y_max)>0.)

def clean_boxes_from_overbound(box_list, scale, img_shape):
    out_dict= {}
    good_boxes = []
    for bbox in box_list['bboxes']:
        if (bbox['bottomright'][0]-bbox['topleft'][0])*(bbox['bottomright'][1]-bbox['topleft'][1])>0 and check_scaled_box(bbox, scale, img_shape):
            good_boxes.append(bbox)
    out_dict['n_boxes'] = len(good_boxes)
    out_dict['bboxes'] = good_boxes
    return out_dict

def convert_dict_to_xml_style(img_filename, bboxes, folder='folder_1', size=None):
    objects = []
    for bbox in bboxes['bboxes']:
        obj_dict = {}
        obj_dict['name'] = 'person'
        obj_dict['pose'] = 'Left'
        obj_dict['truncated'] = 0
        obj_dict['difficult'] = 0
        obj_dict['bndbox'] = {'xmin': bbox['topleft'][0],
                              'ymin': bbox['topleft'][1],
                              'xmax': bbox['bottomright'][0],
                              'ymax': bbox['bottomright'][1]}
        objects.append(obj_dict)
    sizes = {'width': size[0], 'height': size[1], 'depth': size[2]}
    data = {}
    data['folder'] = folder
    data['filename'] = img_filename
    data['size'] = sizes
    data['object'] = objects
    return {'annotation': data}

def make_xml_from_dict(annotations_dict, xml_path, img_shape):
	if not os.path.exists(xml_path):
		os.mkdir(xml_path)

	for k,v in annotations_dict.items():
	    img_filename = ntpath.basename(k)
	    xml_filename = os.path.splitext(img_filename)[0]+'.xml'
	    styled_dict = convert_dict_to_xml_style(img_filename, v, size=img_shape)
	    xml_data = dict2xml(styled_dict)
	    with open(os.path.join(xml_path, xml_filename), 'w') as f:
	        f.write(xml_data)

def make_xml_annotations_from_json(json_source_path, json_files, xml_root, img_shape, split=None, copy_to_root=False):
	data = {}
	for json_file in json_files:
		_data = json.loads(open(os.path.join(json_source_path,json_file), 'r').read())
		for k, v in _data.items():
			if v['n_boxes'] > 0 and all_boxes_nonzero(v['bboxes']):
				data[k] = v
	if not split is None:
		train_keys = np.random.choice(list(data.keys()), int(len(data.keys())*split), replace=False)
		train_data = {k:v for k,v in data.items() if k in train_keys}
		test_data = {k:v for k,v in data.items() if k not in train_keys}
		make_xml_from_dict(train_data, os.path.join(xml_root, 'train'), img_shape)
		make_xml_from_dict(test_data, os.path.join(xml_root, 'test'), img_shape)
		if copy_to_root:
			make_xml_from_dict(data, xml_root, img_shape)	
	else:
		make_xml_from_dict(data, xml_root, img_shape)
