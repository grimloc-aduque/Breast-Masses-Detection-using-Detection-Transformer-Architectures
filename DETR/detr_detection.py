
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import ImageDraw, ImageFont


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1).tolist()


def prepare_for_coco_detection(predictions):
    coco_results = []
    for image_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()
        boxes = prediction["boxes"]
        boxes =  convert_to_xywh(boxes)

        coco_results.extend(
            [
                {
                    "image_id": image_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results


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
        draw.rectangle((x, y, xf, yf), outline='red', width=1)
        text = f'{np.round(score, 2)} - {id2label[label]}'
        draw.text((x, y), text, fill='red', 
                  font=ImageFont.truetype("arial.ttf", 20))

    plt.figure(figsize=(4,4))
    plt.axis("off")
    plt.imshow(image)