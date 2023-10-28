import os
import shutil

import numpy as np
from sklearn.model_selection import KFold
from ultralytics_config import Config


def copy_images(files, data_dir):
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)

    images_folder = f'{data_dir}/images'
    labels_folder = f'{data_dir}/labels'
    os.mkdir(images_folder)
    os.mkdir(labels_folder)
    for file in files:
        shutil.copy(f'./{Config.DATASET}/train_valid/images/{file}.jpg', f'{images_folder}/{file}.jpg')
        shutil.copy(f'./{Config.DATASET}/train_valid/labels/{file}.txt', f'{labels_folder}/{file}.txt')


def copy_train_val_images(train_files, valid_files, fold):
    
    names = ['Mass', 'No-Mass']
    
    data_yaml = f'''
path: C:/Users/Alejandro Duque/Documents/USFQ/Proyecto Integrador/Workspace/{Config.ROOT}/{Config.DATASET}

train: ./fold_{fold}/train/images
val: ./fold_{fold}/valid/images
test: ./test/images

nc: {Config.NUM_CLASSES}
names: {names[:Config.NUM_CLASSES]}
    '''

    data_docker_yaml = f'''
path: /home/{Config.ROOT}/{Config.DATASET}

train: ./fold_{fold}/train/images
val: ./fold_{fold}/valid/images
test: ./test/images

nc: {Config.NUM_CLASSES}
names: {names[:Config.NUM_CLASSES]}
'''
    
    fold_dir =   f'./{Config.DATASET}/fold_{fold}'
    train_dir = f'{fold_dir}/train'
    valid_dir = f'{fold_dir}/valid'

    copy_images(train_files, train_dir)
    copy_images(valid_files, valid_dir)

    open(f'{fold_dir}/data.yaml', 'w+').write(data_yaml)
    open(f'{fold_dir}/data_docker.yaml', 'w+').write(data_docker_yaml)


if __name__ == '__main__':
    train_valid_files = os.listdir(f'./{Config.DATASET}/train_valid/images')
    train_valid_files = [os.path.splitext(file)[0] for file in train_valid_files]

    kfold = KFold(n_splits=10, shuffle=True, random_state=123456)

    for i, (train_index, valid_index) in enumerate(kfold.split(train_valid_files)):
        fold = i+1
        train_files = np.array(train_valid_files)[train_index]
        valid_files = np.array(train_valid_files)[valid_index]
        copy_train_val_images(train_files, valid_files, fold)