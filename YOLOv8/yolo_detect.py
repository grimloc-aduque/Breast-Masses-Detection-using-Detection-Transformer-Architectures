

import cv2
import matplotlib.pyplot as plt


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