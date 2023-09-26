
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageDraw, ImageFont


def plot_annotations(image, annotations, id2label):
    draw = ImageDraw.Draw(image, "RGB")

    for annotation in annotations:
        box = annotation['bbox']
        class_idx = annotation['category_id']
        x,y,w,h = tuple(box)
        draw.rectangle((x,y,x+w,y+h), outline='red', width=2)
        draw.text((x, y), id2label[class_idx], fill='black', 
                font=ImageFont.truetype("arial.ttf", 20))

    plt.figure(figsize=(4,4))
    plt.axis("off")
    plt.imshow(image)


def plot_results(image, results, id2label):
    scores, labels, boxes = results['scores'], results['labels'], results['boxes']

    draw = ImageDraw.Draw(image, "RGBA")

    for score, label, (x, y, xf, yf) in zip(scores.tolist(), labels.tolist(), boxes.tolist()):
        # print(f'Score: {score}')
        draw.rectangle((x, y, xf, yf), outline='red', width=1)
        text = f'{np.round(score, 2)} - {id2label[label]}'
        draw.text((x, y), text, fill='black', 
                  font=ImageFont.truetype("arial.ttf", 18))

    plt.figure(figsize=(4,4))
    plt.axis("off")
    plt.imshow(image)