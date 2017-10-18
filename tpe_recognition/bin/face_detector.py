from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from scipy import misc
import tensorflow as tf
import os
from facenet.src.align.detect_face import *
import numpy as np
import cv2
from datetime import datetime
from keras import backend as K

K.set_image_dim_ordering('th')

def safe_write_image(img, dirname, filename, extention='.jpg'):
# сохраняет изображение img в папку dirname под именем filename с расширением extention. Если такой файл уже есть, то
# добавляет к этому имени '_n' - номер [0..99]
    double_counter = 0
    MAX_TRY = 100
    try_name = os.path.join(dirname,filename)+extention
    for _ in range(MAX_TRY):
        if os.path.isfile(try_name):
            try_name = os.path.join(dirname,filename)+'_'+str(double_counter)+extention
        else:
            cv2.imwrite(try_name, img)
            return
        
def dump_images_from_trackers(saving_dir, trackers, img, frame_counter, dump_interval, max_images_per_id):
# вырезает из кадра изображения объектов и сохраняет их в директорию saving_dir
# saving_dir - директория для сохранения
# trackers - список трекеров
# img - текущий кадр
# frame_counter - счетчик кадров
# dump_interval - сохранение происходит каждые dump_interval кадров
# max_images_per_id - макс.кол-во изображений для каждого пользователя
    if frame_counter % dump_interval == 0:
        for p_id, t_in, tracker, box, t_out in trackers:
            if tracker:
                folder_name = os.path.join(saving_dir, str(p_id))
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                num_of_files = len([x for x in os.listdir(folder_name) if x.endswith('.jpg')])
                if num_of_files < max_images_per_id:
                    image_counter = num_of_files + 1
                    x,y,w,h = box
                    person_img = img[int(y):int(y+h), int(x):int(x+w)]
                    img_name = "id_%d_%d" %(p_id, image_counter)
                    safe_write_image(person_img, folder_name, img_name, extention='.jpg')
                    
def dump_trackers_list(saving_dir, log_file, trackers):
# сохраняет список трекеров в log-файл
# saving_dir - директория для сохранения
# log_file - имя файла
# trackers - список трекеров
    fullname = os.path.join(saving_dir, log_file)
    with open(fullname, 'w') as f:
        for tracker in trackers:
            f.write(str(tracker)+'\n')
            
def draw_boxes(img, trackers):
# рисует трекеры
    for p_id, t_in, tracker, box, t_out in trackers:
        if tracker:
            (x,y,w,h) = map(int, box)
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0), 2)
            cv2.putText(img, 'id:' +str(p_id),(x,y),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0))
    cv2.imshow('ex', img)
    
def find_nearest_box(box, other_boxes, limit):
# находит ближайший к текущему ограничивающему прямоугольнику ближайший в пределах limit
# box (x,y,w,h) - текущий прямоугольник
# other_boxes [(x1,y1,w1,h1), ...]- прямоугольники, среди которых ведется поиск
# limit (x_max,y_max) - диапазон координат, в котором ведется поиск

    x1, y1, w, h = box
    center_x = x1 + w*0.5
    center_y = y1 +h*0.5
    lim_x, lim_y = limit
    nearest_box = ()
    search_ok = False
    dx_min = 1e+20
    dy_min = 1e+20
    for (x1, y1, w, h) in other_boxes:
        another_center_x = x1 + 0.5*w
        another_center_y = y1 + 0.5*h
        dx = abs(center_x - another_center_x)
        dy = abs(center_y - another_center_y)
        if dx <= min(dx_min, lim_x) and dy <= min(dy_min, lim_y):
            dx_min = dx
            dy_min = dy
            nearest_box = (x1, y1, w, h)
            search_ok = True

    return search_ok, nearest_box

def format_boxes(bounding_boxes):
# привидит ограничивающие прямоугольники из формата (x1,y1,x2,y2,acc) в формат(x,y,w,h)
    out = []
    for (x1, y1, x2, y2, acc) in bounding_boxes:
        out.append((x1, y1, x2-x1, y2-y1))
    return out

def rescue_tracker(tracker_, bounding_boxes, img, limit, use_tracker=cv2.TrackerMedianFlow_create):
# пытается восстановить упавший трекер с помощью новой детекции объектов
# tracker_ - упавший трекер
# bounding_boxes - новая детекция объектов
# img - текущий кадр
# limit - диапазон поиска
# use_tracker - тип трекера {cv2.TrackerMedianFlow_create, ...}
    p_id, t_in, tracker, box, t_out = tracker_
    print('trying to rescue tracker for p_id {}'.format(p_id))
    # ищем кандидата - ближайший вновь детектированный box в bounding_boxes, ближайший к последнему моменту жизни трекера
    search_ok, nearest_box = find_nearest_box(box, bounding_boxes, limit)  
    if search_ok:
        # если нашли, то переинициализируем данный трекер
        print('success')
        box = nearest_box
        del tracker
        tracker = use_tracker()
        tracker.init(img, box)
    else:
        print('failed')
        # если нет, то считаем, что объект ушел
        del tracker
        tracker = None
        box = None
        t_out = str(datetime.now())
    return (p_id, t_in, tracker, box, t_out)

def update_trackers_with_boxes(trackers_, bounding_boxes, img, limit, use_tracker=cv2.TrackerMedianFlow_create):
# обновляет трекеры новой детекцией объектов
    trackers = trackers_
    for i in range(0,len(trackers)):
        p_id, t_in, tracker, box, t_out = trackers[i]
        if tracker:
            search_ok, nearest_box = find_nearest_box(box, bounding_boxes, limit)
            if search_ok:
                box = nearest_box
                del tracker
                tracker = use_tracker()
                tracker.init(img, box)
            # если подходящего bounding box не найдено, но трекер жив, то не трогаем трекер
            trackers[i] = (p_id, t_in, tracker, box, t_out)
    return trackers

def update_trackers(
    trackers_, img, limit, img_rgb, minsize, pnet, rnet, onet, threshold, factor, use_tracker=cv2.TrackerMedianFlow_create
):
# обновляет трекеры на каждом кадре
    trackers = trackers_
    status = []
    for i in range(0,len(trackers)):
        p_id, t_in, tracker, box, t_out = trackers[i]
        if tracker:
            # пытаемся обновить трекеры стандартным методом
            ok, box_upd = tracker.update(img)
            if ok:
                trackers[i] = (p_id, t_in, tracker, box_upd, t_out)
            else:
                # пытаемся высстановить упавший трекер новой детекцией
                bounding_boxes, _ = detect_face(img_rgb, minsize, pnet, rnet, onet, threshold, factor)
                bounding_boxes = format_boxes(bounding_boxes)
                trackers[i] = rescue_tracker(trackers[i], bounding_boxes, img, limit, use_tracker=use_tracker)
                # если трекер не восстановлен, то tracker=None
    return trackers

def check_for_new_objects(img, trackers_, bounding_boxes, overlap, p_id_counter, use_tracker=cv2.TrackerMedianFlow_create):
# ищем новые объекты на кадре
    trackers = trackers_
    p_id = p_id_counter
    existing_boxes = []
    if len(trackers)>0:
        for _, __, ___, box, ____ in trackers:
            if box:
                existing_boxes.append(box)
            
        for box in bounding_boxes:
            box_in_trackers, _ = find_nearest_box(box, existing_boxes, overlap)
            
            if not box_in_trackers:
                p_id += 1
                t_in = str(datetime.now())
                tracker = use_tracker()
                tracker.init(img, box)
                trackers.append((p_id,t_in,tracker, box, None)) #заносим его в лист трекеров
    else:
        for box in bounding_boxes:
            p_id += 1
            t_in = str(datetime.now())
            tracker = use_tracker()
            tracker.init(img, box)
            trackers.append((p_id,t_in,tracker, box, None)) #заносим его в лист трекеров
    return p_id, trackers


def get_people_from_video(
    video_file, saving_dir=None, log_file_name='log.txt', tracking_limit=(50,50), detection_frame_step=40,
    tracker_type=cv2.TrackerMedianFlow_create, saving_frame_step=40, max_images_per_id=60, detector_min_size=50,
    threshold=[0.6, 0.7, 0.7], factor=0.7, gpu_memory_fraction=1.
):
# детекция людей на видеозаписи и сохранение их изображений в папке saving_dir
# video_file - исходное видео
# saving_dir - корневая папка для сохранения изображений людей
# log_file_name - имя лог-файла
# tracking_limit - максимальное значение отступов для поиска объекта при потере трекера
# detection_frame_step - интервал кадров для обновления детекции 
# tracker_type - тип трекера
# saving_frame_step - интервал кадров для сохранения изображений
# max_images_per_id - максимальное кол-во изображений одного человека для сохранения
# detector_min_size - минимальный размер изображения для детекции
# threshold - параметр модели распознавания
# factor - параметр модели распознавания
# gpu_memory_fraction - параметр модели распознавания

    camera = cv2.VideoCapture(video_file)
    trackers = [] # инициализация списка трекеров. Каждый объект списка - кортеж (p_id, t_in, tracker, box, t_out)
                  # p_id - идентификатор человека
                  # t_in - время появления человека
                  # tracker - трекер объекта. Если трекер упал, tracker == None
                  # box - текущий ограничивающий прямоугольник объекта
                  # t_out - время потери объекта (пока трекер работает, t_out == None)
    p_id_counter = 0 # счетчик id персон
    frame_counter = 0  # счетчик кадров
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    
        with sess.as_default():
            pnet, rnet, onet = create_mtcnn(sess, None)

        while(1):
            cam_ok, img = camera.read()
            if not cam_ok:
                break

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # обновляем трекеры
            trackers = update_trackers(
                trackers, img,  tracking_limit, img_rgb, detector_min_size,
                pnet, rnet, onet, threshold, factor, use_tracker=tracker_type)

            if frame_counter % detection_frame_step == 0 or len(trackers) == 0:
                # на первом шаге или на каждом frame_step - шаге обновляем трекеры новой детекцией:
                bounding_boxes, _ = detect_face(
                    img_rgb, detector_min_size, pnet, rnet, onet, threshold, factor)
                #print(bounding_boxes)
                bounding_boxes = format_boxes(bounding_boxes)
                trackers = update_trackers_with_boxes(trackers, bounding_boxes, img, tracking_limit, use_tracker=tracker_type)
                #print(bounding_boxes)
                p_id_counter, trackers = check_for_new_objects(
                    img, trackers, bounding_boxes, tracking_limit, p_id_counter, use_tracker=tracker_type) #проверяем, есть ли новые объекты
                #print(trackers)

            dump_images_from_trackers(saving_dir, trackers, img, frame_counter, saving_frame_step, max_images_per_id) # сохраняем изображения объектов в папке saving_dir
            draw_boxes(img, trackers)
            interrupt = cv2.waitKey(10)

            if interrupt & 0xFF == ord('q'):
                print(frame_counter)
                break
            frame_counter += 1

    dump_trackers_list(saving_dir, log_file_name, trackers)
    cv2.destroyAllWindows()
    camera.release()