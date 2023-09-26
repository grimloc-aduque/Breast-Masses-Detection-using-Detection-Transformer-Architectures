# %%
import json
import os
import shutil

import numpy as np
from detr_config import Config
from sklearn.model_selection import KFold

# %%
def copy_images(files, data_dir):
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)

    for file in files:
        shutil.copy(f'./{Config.DATASET}/train_valid/{file}', f'{data_dir}/{file}')
    

def annotations_dict(full_annotations, files):
    images = [
        img for img in full_annotations['images'] 
        if img['file_name'] in files
    ]

    image_ids = [
        img['id'] for img in images
    ]

    annotations = [
        annotation for annotation in full_annotations['annotations'] 
        if annotation['image_id'] in image_ids
    ]
    
    categories = [
        {
            "id": 0,
            "name": "Mass",
            "supercategory": "none"
        },
        {
            "id": 1,
            "name": "No-Mass",
            "supercategory": "none"
        }
    ]
    
    categories[:Config.NUM_CLASSES]

    annotations_dict = {
        'categories': categories[:Config.NUM_CLASSES],
        'images': images,
        'annotations': annotations
    }
    
    return annotations_dict


def save_annotations(annotations_dict, fold_dir):
    annotations_str = json.dumps(annotations_dict, indent=4)
    with open(f'{fold_dir}/_annotations.coco.json', 'w+') as f:
        f.write(annotations_str)


# %%
train_valid_files = os.listdir(f'{Config.DATASET}/train_valid')
train_valid_files = [file for file in train_valid_files if 'annotations' not in file]
train_valid_anotations = json.load(open(f'{Config.DATASET}/train_valid/_annotations.coco.json'))


kfold = KFold(n_splits=10, shuffle=True, random_state=123456)

for i, (train_index, valid_index) in enumerate(kfold.split(train_valid_files)):
    fold = i+1
    
    fold_dir =   f'./{Config.DATASET}/fold_{fold}'
    train_dir = f'{fold_dir}/train'
    valid_dir = f'{fold_dir}/valid'
    
    train_files = np.array(train_valid_files)[train_index]
    copy_images(train_files, train_dir)
    train_annotations_json = annotations_dict(train_valid_anotations, train_files)
    save_annotations(train_annotations_json, train_dir)
    
    valid_files = np.array(train_valid_files)[valid_index]
    copy_images(valid_files, valid_dir)
    valid_annotations_json = annotations_dict(train_valid_anotations, valid_files)
    save_annotations(valid_annotations_json, valid_dir)
    
    

# %%



