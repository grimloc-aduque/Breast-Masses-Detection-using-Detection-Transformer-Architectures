
from ultralytics import YOLO
from yolo_config import Config
import os

if __name__ == '__main__':
    
    dataset = 'InBreast_Yolov8'
    model_sizes = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']


    for model_size in model_sizes:
        for fold in range(1,11):

            fold_dir = f'fold_{fold}'
            dataset_dir = os.path.join(Config.DATASET, fold_dir)
            yaml_path = os.path.join(dataset_dir, Config.YAML_FILE)

            train_project = os.path.join(Config.RUNS_DIR, Config.DATASET, model_size, fold_dir)

            model = YOLO(f"./Weights/{model_size}")
            model.train(
                data = yaml_path,
                epochs = Config.EPOCHS,
                batch = Config.BATCH_SIZE,
                project = train_project)
            
            break
        break
    