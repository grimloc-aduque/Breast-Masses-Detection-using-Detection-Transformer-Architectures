

import cv2
import matplotlib.pyplot as plt
from yolo_config import Config
import os
import pandas as pd


def show_image(training_dir, file_name):
    try:
        img = cv2.imread(f'{training_dir}/{file_name}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(5,5), dpi=120)
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Image not available")
        print(e)
        pass


def plot_detection_comparison(training_dir):
    plt.figure(figsize=(5,10), dpi=200)

    plt.subplot(1, 2, 1)
    img = cv2.imread(f'{training_dir}/val_batch0_labels.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.title("Labels", fontsize=5)
    plt.axis('off')
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    img = cv2.imread(f'{training_dir}/val_batch0_pred.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.title("Detections", fontsize=5)
    plt.axis('off')
    plt.imshow(img)

    plt.show()