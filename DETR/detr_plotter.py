
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
from detr_file_manager import FileManager
from detr_metrics import MetricsAggregator, metrics_names
from PIL import ImageDraw, ImageFont


class Plotter:
    
    def __init__(self, file_manager:FileManager, metrics_aggregator:MetricsAggregator):
        self.file_manager = file_manager
        self.metrics_aggregator = metrics_aggregator

    def min_max(self, pixel_values):
        return (pixel_values - pixel_values.min()) / (pixel_values.max() - pixel_values.min())


    def plot_annotations(self, pixel_values, annotations, id2label, ax=None):
        pixel_values = self.min_max(pixel_values)
        image = F.to_pil_image(pixel_values)
        draw = ImageDraw.Draw(image, "RGB")
        
        class_labels = annotations['class_labels']
        boxes = annotations['boxes']
        for label, box in zip(class_labels, boxes):
            box = box * 800
            (xc, yc, w, h) = tuple(box)
            x_min = xc - w/2
            y_min = yc - h/2
            draw.rectangle(
                xy=(x_min, y_min, x_min+w, y_min+h), 
                outline='red', 
                width=5
            )
            draw.text(
                xy = (x_min, y_min), 
                text = id2label[label.item()], 
                fill = 'black', 
                font = ImageFont.load_default()
            )
            
        if ax == None:
            fig, ax = plt.subplots(figsize=(3, 3))
        ax.axis("off")
        ax.imshow(image)


    def plot_predictions(self, pixel_values, predictions, id2label, ax=None):
        pixel_values = self.min_max(pixel_values)
        image = F.to_pil_image(pixel_values)
        draw = ImageDraw.Draw(image, "RGBA")

        scores = predictions['scores']
        labels = predictions['labels']
        boxes = predictions['boxes']
        for score, label, box in zip(scores.tolist(), labels.tolist(), boxes.tolist()):
            (x_min, y_min, x_max, y_max) = box
            draw.rectangle(
                xy=(x_min, y_min, x_max, y_max), 
                outline='green', 
                width=5
            )
            text = f'{np.round(score, 2)} - {id2label[label]}'
            draw.text(
                xy=(x_min, y_min), 
                text=text, 
                fill='black', 
                font = ImageFont.load_default()
            )

        if ax == None:
            fig, ax = plt.subplots(figsize=(3, 3))
        ax.axis("off")
        ax.imshow(image)
        

    def plot_comparison(self, pixel_values, annotations, predictions, id2label, axs=[]):
        if len(axs)==0:
            f, axs = plt.subplots(1,2)
            axs = axs.flatten()
        self.plot_annotations(pixel_values, annotations, id2label, ax=axs[0])
        self.plot_predictions(pixel_values, predictions, id2label, ax=axs[1])
        
        

    def plot_batch_comparison(self, batch_predictions, dataset, threshold, batch_id, 
                              show=False, save_fig = True):
        nrows = 2
        ncols = len(batch_predictions)

        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols, nrows), dpi=80*ncols)
        for i, (image_id, predictions) in enumerate(batch_predictions.items()):
            pixel_values, annotations = dataset.__getitem__(image_id)
            self.plot_comparison(pixel_values, annotations, predictions, {0:'Mass'}, axs=axs[:,i])

        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0.01)
        
        if show:
            plt.show()
        if save_fig:
            plot_path = self.file_manager.get_detection_plot_path(threshold, batch_id)
            plt.savefig(plot_path)
        plt.close()



    def plot_metrics(self):
        metrics = self.metrics_aggregator.load_metrics()
        fig, axes = plt.subplots(1, 2, figsize=(9,4), dpi=250)
        fig.tight_layout(pad=1.5)
        axes = axes.flat

        for i, metric in enumerate(metrics_names[0:2]):
            metrics_name = metric.split("|")[0]
            ax = axes[i]
            ax.plot(metrics.index, metrics[metric], alpha=0.6)
            ax.scatter(metrics.index, metrics[metric], alpha=0.6, s=10)

            y_ticks = np.round(np.arange(0,1.1,0.2), 1)
            x_ticks = metrics.index
            ax.set_yticks(y_ticks, labels=y_ticks, fontsize=6)
            ax.set_xticks(x_ticks, labels=x_ticks, fontsize=6)

            ax.set_xlabel("Detection Threshold", fontsize=8)
            ax.set_ylabel(metrics_name, fontsize=8)
            
        plot_path = self.file_manager.get_metrics_plot_path()
        plt.savefig(plot_path)
        plt.close()

