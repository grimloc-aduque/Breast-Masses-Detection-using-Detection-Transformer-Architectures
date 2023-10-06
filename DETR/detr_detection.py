
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
from PIL import ImageDraw, ImageFont


def min_max(pixel_values):
    return (pixel_values - pixel_values.min()) / (pixel_values.max() - pixel_values.min())


def plot_annotations(pixel_values, annotations, id2label, ax=None):
    pixel_values = min_max(pixel_values)
    image = F.to_pil_image(pixel_values)
    draw = ImageDraw.Draw(image, "RGB")
    
    class_labels = annotations['class_labels']
    boxes = annotations['boxes']
    for label, box in zip(class_labels, boxes):
        box = box * 800
        (x_center, y_center, w, h) = tuple(box)
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


def plot_predictions(pixel_values, predictions, id2label, ax=None):
    pixel_values = min_max(pixel_values)
    image = F.to_pil_image(pixel_values)
    draw = ImageDraw.Draw(image, "RGBA")

    scores = predictions['scores']
    labels = predictions['labels']
    boxes = predictions['boxes']
    for score, label, box in zip(scores.tolist(), labels.tolist(), boxes.tolist()):
        (x_min, y_min, x_max, y_max) = box
        draw.rectangle(
            xy=(x_min, y_min, x_max, y_max), 
            outline='red', 
            width=3
        )
        text = f'{np.round(score, 2)} - {id2label[label]}'
        draw.text(
            xy=(x_min, y_min), 
            text=text, 
            fill='black', 
            font=ImageFont.truetype("arial.ttf", 22)
        )

    if ax == None:
        fig, ax = plt.subplots(figsize=(3, 3))
    ax.axis("off")
    ax.imshow(image)
    

def plot_comparison(pixel_values, annotations, predictions, id2label):
    f, axs = plt.subplots(1,2)
    axs = axs.flatten()
    plot_annotations(pixel_values, annotations, id2label, ax=axs[0])
    plot_predictions(pixel_values, predictions, id2label, ax=axs[1])

