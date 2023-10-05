
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
from PIL import ImageDraw, ImageFont


def min_max(image):
    return (image - image.min()) / (image.max() - image.min())

def plot_annotations(image, target, id2label, ax=None):
    image = min_max(image)
    image = F.to_pil_image(image)
    draw = ImageDraw.Draw(image, "RGB")
    
    class_labels = target['class_labels']
    boxes = target['boxes']
    for label, box in zip(class_labels, boxes):
        box = box * 800
        x_center, y_center, w, h = tuple(box)
        x = x_center - w/2
        xf = x_center + w/2
        y = y_center - h/2
        yf = y_center + h/2     
        draw.rectangle((x,y,xf,yf), outline='red', width=3)
        draw.text(
            xy = (x, y), 
            text = id2label[label.item()], 
            fill = 'black', 
            font = ImageFont.truetype("arial.ttf", 22)
        )
        
    if ax == None:
        fig, ax = plt.subplots(figsize=(3, 3))
    ax.axis("off")
    ax.imshow(image)


def plot_predictions(image, predictions, id2label, ax=None):
    image = min_max(image)
    image = F.to_pil_image(image)
    draw = ImageDraw.Draw(image, "RGBA")

    scores = predictions['scores']
    labels = predictions['labels']
    boxes = predictions['boxes']
    for score, label, box in zip(scores.tolist(), labels.tolist(), boxes.tolist()):
        (x, y, xf, yf) = box
        draw.rectangle((x, y, xf, yf), outline='red', width=3)
        text = f'{np.round(score, 2)} - {id2label[label]}'
        draw.text((x, y), text, fill='black', 
                  font=ImageFont.truetype("arial.ttf", 22))

    if ax == None:
        fig, ax = plt.subplots(figsize=(3, 3))
    ax.axis("off")
    ax.imshow(image)
    

def plot_comparison(image, annotations, predictions, id2label):
    f, axs = plt.subplots(1,2)
    axs = axs.flatten()
    plot_annotations(image, annotations, id2label, ax=axs[0])
    plot_predictions(image, predictions, id2label, ax=axs[1])