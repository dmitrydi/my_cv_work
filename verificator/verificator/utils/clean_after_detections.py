import os
import numpy as np
import cv2

def clean_dirs(root_path, clean_mode='is_empty', min_aspect_ratio=1):
# удаление из подпапок root_path пустых папок и изображений, у которых (h/w) < min_aspect_ratio
  for dirname in os.listdir(root_path):
    if clean_mode == 'with_aspect':
      for fimg in os.listdir(os.path.join(root_path,dirname)):
        fimg_full = os.path.join(root_path, dirname, fimg)
        img=cv2.imread(fimg_full)
        if (img is not None) and img.shape[1]>0:
          aspect = img.shape[0]/img.shape[1]
          if aspect < min_aspect_ratio:
            print('removing img {}'.format(fimg_full))
            os.remove(fimg_full)
        else:
          print('removing img {}'.format(fimg_full))
          os.remove(fimg_full)

    if not os.listdir(os.path.join(root_path, dirname)):
      print('removing dir {}'.format(os.path.join(root_path, dirname)))
      os.rmdir(os.path.join(root_path, dirname))

def rename_folders(root_path, start_number, exclude=None):
# переименование подпапок root_path по порядку начиная с номера start_number
  folders = os.listdir(root_path)
  numb = start_number
  # safe rename
  for folder in folders:
    fname = folder+'_'
    os.rename(os.path.join(root_path, folder), os.path.join(root_path, fname))
  # main rename
  folders = os.listdir(root_path)
  for folder in folders:
    fname = str(numb)
    new_name = os.path.join(root_path, fname)
    os.rename(os.path.join(root_path, folder), new_name)
    safe_rename_in_dir(new_name)
    print('{} renamed to {}'.format(folder, os.path.basename(new_name)))
    numb += 1

def safe_rename_in_dir(folder_name):
# переименоване файлов в папке floder_name в формат floder_name_1.ext, folder_name_2.ext, ...
  files = os.listdir(folder_name)
  for fname in files:
    new_fname = os.path.splitext(fname)[0]+'___'+os.path.splitext(fname)[1]
    os.rename(os.path.join(folder_name, fname), os.path.join(folder_name, new_fname))
  numb = 1
  files = os.listdir(folder_name)
  for fname in files:
    new_name = os.path.basename(folder_name)+'_'+str(numb)+os.path.splitext(fname)[1]
    os.rename(os.path.join(folder_name, fname), os.path.join(folder_name, new_name))
    numb += 1

def move_files_to_test_dir_for_verificator(root_name, test_dir, min_samples_in_source=15, num_files_to_move = 5):
# перемещает для каждой подапки root_name изображения в test_dir если кол-во изображений в root_name/folder >= min_sampples_in_source
# num_files_to_move - сколько фалов перемещаем
# root_name, test_dir - должны быть указаны абсолютные пути
  def select_files_to_move(list_of_files, num_files_to_move):
    files = np.array(list_of_files)
    return  list(np.random.choice(files, num_files_to_move, replace=False))

  def move_files(files, src_dir, dst_dir):
    if not os.path.exists(dst_dir):
      os.mkdir(dst_dir)
    for file_ in files:
      os.rename(os.path.join(src_dir, file_), os.path.join(dst_dir, file_))

  fcntr = 0
  dirs = os.listdir(root_name)
  if os.path.basename(test_dir) in dirs:
    dirs.remove(os.path.basename(test_dir))
  for dir_ in dirs:
    src_dir = os.path.join(root_name, dir_)
    dir_files = os.listdir(src_dir)
    if len(dir_files) >= min_samples_in_source:
      files_to_move = select_files_to_move(dir_files, num_files_to_move)
      fcntr += len(files_to_move)
      dst_dir = os.path.join(test_dir, dir_)
      move_files(files_to_move, src_dir, dst_dir)
  print('{} files moved'.format(fcntr))
