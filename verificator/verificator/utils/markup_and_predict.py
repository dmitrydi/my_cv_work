from .rect_calc import IoU
import cv2
import json
import os
from darkflow.net.build import TFNet

def hand_mark_boxes(source_path, data_path,
               data_filename='markup.json', log_filename='log.txt',
               extentions_list=['jpg','JPG','jpeg','JPEG']):
    # перебирает изображения в папке source_path, для каждого изображения выделяются bboxes для объектов
    # разметка сохраняется в файле data_path/data_filename.json
    # список обработанных изображений сохраняется в файле data_path/log_filename
    # extentions_list - список расширений изображений для обработки
    if not os.path.exists(source_path):
        raise OSError('source path does not exist')

    if not os.path.exists(data_path):
        os.makedirs(data_path)
        
    cut_files = [] #уже обработанные изображения в source_path
    if os.path.isfile(os.path.join(data_path,log_filename)):
        with open(os.path.join(data_path,log_filename), 'r') as f:
            content = f.readlines()
            cut_files = [os.path.basename(x.rstrip()) for x in content]
            
    with open(os.path.join(data_path,log_filename), 'a') as f:
        fcounter = 0
        from_center = False
        data_dict=dict() #словарь для хранения разметки

        for file in os.listdir(source_path):
            flag = False

            for ext in extentions_list:
                flag = flag or file.endswith(ext)

            if (not file in cut_files) and flag:
                img = cv2.imread(os.path.join(source_path, file))
                n_obj = 0 # кол-во объектов на изображении
                bbox_list = [] # список ббоксов на изображении

                while(True):
                    r = cv2.selectROI('win_roi', img, from_center)
                    if r == (0,0,0,0):
                        break
                    topleft=tuple([int(r[0]), int(r[1])])
                    bottomright=tuple([int(r[0]+r[2]), int(r[1]+r[3])])
                    bbox={'topleft': topleft, 'bottomright': bottomright}
                    bbox_list.append(bbox)
                    n_obj += 1
                    
                f.write(os.path.join(source_path,file)+'\n')
                fdict={'n_boxes': n_obj, 'bboxes':bbox_list}
                data_dict[os.path.join(source_path,file)] = fdict

            interrupt = cv2.waitKey()

            if interrupt & 0xFF == ord('q'):
                break
                                
    cv2.destroyAllWindows()
    
    with open(os.path.join(data_path, data_filename), 'w') as outfile:
        json.dump(data_dict, outfile)

def darkflow_predict_boxes_from_model(tfnet, source_path, data_path, filelist = None,
                          data_filename='markup_darkflow.json', log_filename='log_darkflow.txt',
                          extentions_list=['jpg','JPG','jpeg','JPEG'], read_logs=True, verbose=True,
                          num_files=None):
# предсказывает ббоксы для коллекции изображений в папке source_path
# tfnet - модель darkflow, которая предсказывает ббоксы
# сохраняет предсказания в json-файле в виде словаря
# {'img_name': {'n_obj': N, 'bboxes': [{'topleft': [x,y], 'bottomright':[x, y]}, ...]}, ...}
# source_path - путь к папке с коллекцией изображений
# data_path - путь к папке, куда будут складываться результаты предсказаний
# filelist - список файлов, которые будут обрабатываться
# data_filename - имя файла, куда будут сохраняться предсказания
# log_filename - имя лог-файла, куда будут писаться имена обработанных файлов
# extentions_list - список расширений файлов, которые будут обрабатываться
# read_logs - флаг, если True, то при повторном запуске не будут обрабатываться изображения, уже перечисленные в лог-файле
# num_files - максимальное кол-во файлов в директории source_path, которые будут обрабатываться. Если None, то все
# --------------
# опции модели darknet:
# model - путь к конфигурационному файлу иодели
# weights - путь к весам модели
# config - путь к cfg-директории модели
# gpu - доля использования GPU. Если None, то расчет идет на CPU

    # опции модели darknet
    # options=dict()
    # options['model']=model
    # options['load']=weights
    # if gpu is not None:
    #     options['gpu']=gpu
    # options['threshold']=threshold
    # options['config']=config
    # options['summary']=summary
    # # загрузили модель
    # tfnet = TFNet(options)
    n_files=0
    
    # проверяем, что source_path есть
    if not os.path.exists(source_path):
        raise OSError('source path does not exist')

    # если нет директории data_path, то создаем ее
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        
    cut_files = [] #уже обработанные изображения в source_path
    # если есть лог-файл и мы не хотим овторно обрабатывать файлы, то включаем их в список исключений
    if os.path.isfile(os.path.join(data_path,log_filename)) and read_logs:
        with open(os.path.join(data_path,log_filename)) as f:
            content = f.readlines()
            cut_files = [x.rstrip() for x in content]
    
    # если задан список фалов для обработки, то берем только файлы из него
    if filelist is not None:
        files = [x for x in os.listdir(source_path) if os.path.splitext(x)[0] in filelist]
    else:
        files = os.listdir(source_path)
            
    with open(os.path.join(data_path,log_filename), 'a') as f: # начинаем запись в log-файл
        data_dict=dict()
        
        for file in files: # для всех файлов в директории
            flag = False

            for ext in extentions_list: # если файл имеет нужное расширение, flag == True
                flag = flag or file.endswith(ext)

            if (not file in cut_files) and flag:
                n_files += 1
                img = cv2.imread(os.path.join(source_path, file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # необходимо изображение RGB
                n_obj = 0 # кол-во объектов на изображении
                bbox_list = [] # список ббоксов на изображении
                predicted_boxes = tfnet.return_predict(img) # предсказываем ббоксы на изображении
                
                if verbose:
                    print('file {}'.format(file))
        
                for pred in predicted_boxes: # формирует список словарей из ббоксов
                    if pred['label']=='person':
                        n_obj += 1
                        topleft=tuple([pred['topleft']['x'], pred['topleft']['y']])
                        bottomright=tuple([pred['bottomright']['x'], pred['bottomright']['y']])
                        bbox={'topleft': topleft, 'bottomright': bottomright}
                        bbox_list.append(bbox)
                    
                f.write(file+'\n')
                fdict={'n_boxes': n_obj, 'bboxes':bbox_list}
                data_dict[os.path.join(source_path,file)] = fdict # словарь {'img_name': {'n_obj': N, 'bboxes': [{'topleft': [x,y], 'bottomright':[x, y]}, ...]}, ...}
                
            if num_files is not None:
                if n_files > num_files:
                    break
                
    with open(os.path.join(data_path, data_filename), 'w') as outfile:
        json.dump(data_dict, outfile)

def darkflow_predict_boxes(source_path, data_path,
                          data_filename='markup_darkflow.json', log_filename='log_darkflow.txt',
                          extentions_list=['jpg','JPG','jpeg','JPEG'], read_logs=True, verbose=True,
                          model='../data/yolo_data/cfg/yolo.cfg',
                          weights='../data/yolo_data/bin/yolo.weights',
                          config='../data/yolo_data/cfg/',
                          summary=None,
                          gpu=0.8, threshold=0.5, num_files=None):
# предсказывает ббоксы для коллекции изображений в папке source_path
# сохраняет предсказания в json-файле в виде словаря
# {'img_name': {'n_obj': N, 'bboxes': [{'topleft': [x,y], 'bottomright':[x, y]}, ...]}, ...}
# source_path - путь к папке с коллекцией изображений
# data_path - путь к папке, куда будут складываться результаты предсказаний
# data_filename - имя файла, куда будут сохраняться предсказания
# log_filename - имя лог-файла, куда будут писаться имена обработанных файлов
# extentions_list - список расширений файлов, которые будут обрабатываться
# read_logs - флаг, если True, то при повторном запуске не будут обрабатываться изображения, уже перечисленные в лог-файле
# num_files - максимальное кол-во файлов в директории source_path, которые будут обрабатываться. Если None, то все
# --------------
# опции модели darknet:
# model - путь к конфигурационному файлу иодели
# weights - путь к весам модели
# config - путь к cfg-директории модели
# gpu - доля использования GPU. Если None, то расчет идет на CPU

    # опции модели darknet
    options=dict()
    options['model']=model
    options['load']=weights
    if gpu is not None:
        options['gpu']=gpu
    options['threshold']=threshold
    options['config']=config
    options['summary']=summary
    # загрузили модель
    tfnet = TFNet(options)
    n_files=0
    
    # проверяем, что source_path есть
    if not os.path.exists(source_path):
        raise OSError('source path does not exist')

    # если нет директории data_path, то создаем ее
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        
    cut_files = [] #уже обработанные изображения в source_path
    # если есть лог-файл и мы не хотим овторно обрабатывать файлы, то включаем их в список исключений
    if os.path.isfile(os.path.join(data_path,log_filename)) and read_logs:
        with open(os.path.join(data_path,log_filename)) as f:
            content = f.readlines()
            cut_files = [x.rstrip() for x in content]
            
    with open(os.path.join(data_path,log_filename), 'a') as f: # начинаем запись в log-файл
        data_dict=dict()
        
        for file in os.listdir(source_path): # для всех файлов в директории
            flag = False

            for ext in extentions_list: # если файл имеет нужное расширение, flag == True
                flag = flag or file.endswith(ext)

            if (not file in cut_files) and flag:
                n_files += 1
                img = cv2.imread(os.path.join(source_path, file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # необходимо изображение RGB
                n_obj = 0 # кол-во объектов на изображении
                bbox_list = [] # список ббоксов на изображении
                predicted_boxes = tfnet.return_predict(img) # предсказываем ббоксы на изображении
                
                if verbose:
                    print('file {}'.format(file))
        
                for pred in predicted_boxes: # формирует список словарей из ббоксов
                    if pred['label']=='person':
                        n_obj += 1
                        topleft=tuple([pred['topleft']['x'], pred['topleft']['y']])
                        bottomright=tuple([pred['bottomright']['x'], pred['bottomright']['y']])
                        bbox={'topleft': topleft, 'bottomright': bottomright}
                        bbox_list.append(bbox)
                    
                f.write(file+'\n')
                fdict={'n_boxes': n_obj, 'bboxes':bbox_list}
                data_dict[os.path.join(source_path,file)] = fdict # словарь {'img_name': {'n_obj': N, 'bboxes': [{'topleft': [x,y], 'bottomright':[x, y]}, ...]}, ...}
                
            if num_files is not None:
                if n_files > num_files:
                    break
                
    with open(os.path.join(data_path, data_filename), 'w') as outfile:
        json.dump(data_dict, outfile)

def clean_dict_keys_from_path(in_dict):
    return({ os.path.split(k)[-1]:v for k,v in in_dict.items() })

def detection_scores(gt_markup, det_markup, iou_threshold=0.5):
# считает precision, recall для предсказаных ббоксов по коллекции изображений
# gt_markup - json файл с разметкой
# det_markup - json файл с предсказаниями
# iou_threshold - порог перекрытия предсказанного и gt-ббокса, когда детекция считается верной
    with open(gt_markup, 'r') as f:
        gt_data_ = json.loads(f.read())
        
    with open(det_markup, 'r') as f:
        pred_data_ = json.loads(f.read())

    gt_data = clean_dict_keys_from_path(gt_data_)
    pred_data = clean_dict_keys_from_path(pred_data_)
        
    common_keys = get_common_keys(gt_data, pred_data)

    if len(common_keys) == 0:
        raise ValueError('No common images in files')
    
    HITS_TOT, NO_TOT, NR_TOT = 0., 0., 0.
    
    for img_name in common_keys:
        bboxes_gt = get_boxes_from_dict(gt_data[img_name])
        bboxes_pred = get_boxes_from_dict(pred_data[img_name])
        hits = get_hits(bboxes_gt, bboxes_pred, iou_threshold=iou_threshold)
        #FA, MD = get_fa_md(bboxes_gt, bboxes_pred, iou_threshold=iou_threshold) # FalseAlarms, MissedDetections
        NO = pred_data[img_name]['n_boxes'] # Number of Objects (detected)
        NR = gt_data[img_name]['n_boxes'] # Number of Objects (ground-truth)
        #FA_TOT += FA
        HITS_TOT += hits
        NO_TOT += NO
        NR_TOT += NR
    
    precision = HITS_TOT/(NO_TOT)
    recall = HITS_TOT/NR_TOT
        
    return precision, recall, HITS_TOT, NO_TOT, NR_TOT

def show_rectangles(gt_markup, pred_markup, img_dir, saving_dir, roi, iou_threshold=0.5):
    with open(gt_markup, 'r') as f:
        gt_data_ = json.loads(f.read())
        
    with open(pred_markup, 'r') as f:
        pred_data_ = json.loads(f.read())

    gt_data = clean_dict_keys_from_path(gt_data_)
    pred_data = clean_dict_keys_from_path(pred_data_)
        
    common_keys = get_common_keys(gt_data, pred_data)

    if len(common_keys) == 0:
        raise ValueError('No common images in files')
    
    for img_name in common_keys:
        bboxes_gt = get_boxes_from_dict(gt_data[img_name])
        bboxes_pred = get_boxes_from_dict(pred_data[img_name])
        bboxes_gt_roi = get_boxes_in_roi(bboxes_gt, roi)
        bboxes_pred_roi = get_boxes_in_roi(bboxes_pred, roi)
        img = cv2.imread(os.path.join(img_dir, img_name))
        for x1, y1, x2, y2 in bboxes_gt_roi:
            img = cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0))
        matched_boxes = get_matched(bboxes_gt_roi, bboxes_pred_roi, iou_threshold=iou_threshold)
        for x1, y1, x2, y2 in bboxes_pred_roi:
            img = cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255))
        for x1,y1,x2,y2 in matched_boxes:
            img = cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0))
        cv2.imwrite(os.path.join(saving_dir,'marked_'+img_name), img)



def detection_scores_in_roi(gt_markup, det_markup, roi, iou_threshold=0.5):
    with open(gt_markup, 'r') as f:
        gt_data_ = json.loads(f.read())
        
    with open(det_markup, 'r') as f:
        pred_data_ = json.loads(f.read())

    gt_data = clean_dict_keys_from_path(gt_data_)
    pred_data = clean_dict_keys_from_path(pred_data_)
        
    common_keys = get_common_keys(gt_data, pred_data)

    if len(common_keys) == 0:
        raise ValueError('No common images in files')
    
    HITS_TOT, NO_TOT, NR_TOT = 0., 0., 0.
    
    for img_name in common_keys:
        bboxes_gt = get_boxes_from_dict(gt_data[img_name])
        bboxes_pred = get_boxes_from_dict(pred_data[img_name])
        bboxes_gt_roi = get_boxes_in_roi(bboxes_gt, roi)
        bboxes_pred_roi = get_boxes_in_roi(bboxes_pred, roi)
        NO = len(bboxes_pred_roi)#pred_data[img_name]['n_boxes'] # Number of Objects (detected)
        NR = len(bboxes_gt_roi)#gt_data[img_name]['n_boxes'] # Number of Objects (ground-truth)
        hits = get_hits(bboxes_gt_roi, bboxes_pred_roi, iou_threshold=iou_threshold)
        #FA, MD = get_fa_md(bboxes_gt, bboxes_pred, iou_threshold=iou_threshold) # FalseAlarms, MissedDetections
        #FA_TOT += FA
        HITS_TOT += hits
        NO_TOT += NO
        NR_TOT += NR
    
    precision = HITS_TOT/(NO_TOT )
    recall = HITS_TOT/(NR_TOT)
        
    return precision, recall, HITS_TOT, NO_TOT, NR_TOT

def box_in_roi(box, roi):
    x1, y1, x2, y2 = box
    x1r, y1r, x2r, y2r = roi
    return (x1>=x1r and x2<=x2r and y1>=y1r and y2<=y2r)

def get_boxes_in_roi(boxes, roi):
    ans = [box for box in boxes if box_in_roi(box, roi)]
    return ans

def get_fa_md(gt_list, pred_list, iou_threshold=0.5):
# ищет FalseAlarms и MissedDetections для списка ббоксов gt_list (ground-truth) и предсказаного списка боксов pred_list
# iou_threshold - порог перекрытия истинного и предсказанного ббоксов, когда детекция считается верной
    MD = 0
    FA = 0
    while(len(gt_list)>0 and len(pred_list)>0):
        bbox = gt_list.pop()
        closest_box = find_closest_box(bbox, pred_list, iou_threshold)
        if closest_box == ():
            MD += 1
        else:
            pred_list = [x for x in pred_list if x != closest_box]
    FA = len(pred_list)
    
    return FA, MD

def get_hits(gt_list, pred_list, iou_threshold=0.5):
    hits = 0
    while(len(gt_list) > 0):
        bbox = gt_list.pop()
        closest_box = find_closest_box(bbox, pred_list, iou_threshold)
        if closest_box != ():
            hits += 1
            pred_list = [ x for x in pred_list if x != closest_box ]
    return hits

def get_matched(gt_list, pred_list, iou_threshold=0.5):
    matched = []
    while(len(gt_list) > 0):
        bbox = gt_list.pop()
        closest_box = find_closest_box(bbox, pred_list, iou_threshold)
        if closest_box != ():
            matched.append(closest_box)
            pred_list = [ x for x in pred_list if x != closest_box ]
    return matched

def find_closest_box(ref_box, box_list, iou_threshold):
# ищет ближайший к ref_box bbox в списке box_list по критерию max IoU|IoU >= iou_threshold
# если IoU_max < iou_threshold, то сичтается, что ближайшего бокса нет
    closest_box = ()
    max_iou = iou_threshold
    for box in box_list:
        iou = IoU(ref_box, box)
        if iou >= max_iou:
            max_iou = iou
            closest_box = box
    return closest_box

def get_common_keys(dict1, dict2):
# возвращает список общих ключей в словарях dict1, dict2
    keys1 = dict1.keys()
    keys2 = dict2.keys()
    common = [k for k in keys1 if k in keys2]
    return common

def get_boxes_from_dict(box_dict_):
# парсинг словаря box_dict_ = {'bboxes': [{'topleft': [x1, y1], 'bottomright': [x2, y2]}, ...,]}
# возвращает список [(x1, y1, x2, y2), ...]
    box_dict = box_dict_.copy()
    ans = []
    for bbox in box_dict['bboxes']:
        tl = bbox['topleft']
        br = bbox['bottomright']
        coords = tl + br
        ans.append(tuple(coords))
    return ans
