
import sys
from io import StringIO

import torch
from coco_eval import CocoEvaluator
from colorama import Fore
from faster_config import Config
from faster_metrics import metrics_names
from faster_plotter import Plotter

STDOUT = sys.stdout

class ModelEvaluator:

    def __init__(self, model, plotter:Plotter):
        self.model = model
        self.plotter = plotter

    # Coco Formating

    def generate_predictions(self, batch, image_ids, threshold):
        pixel_values = batch["pixel_values"].to(Config.DEVICE)
        with torch.no_grad():
            outputs = self.model(pixel_values)
        detections = {}
        for i, output in enumerate(outputs):
            image_id = image_ids[i]
            detection = {
                'boxes': [],
                'labels': [],
                'scores': []
            }
            if len(output['boxes']) == 0:
                detection = {k:torch.Tensor(v) for k, v in detection.items()}
                detections[image_id] = detection
                continue
            for i, score in enumerate(output['scores']):
                if score > threshold:
                    detection['boxes'].append(output['boxes'][i])
                    detection['labels'].append(output['labels'][i])
                    detection['scores'].append(output['scores'][i])
                detection = {k:torch.Tensor(v) for k, v in detection.items()}
                detections[image_id] = detection
        return detections # bbox: (x_min, y_min, x_max, y_max)

    def _convert_to_xywh(self, boxes):
        xmin, ymin, xmax, ymax = boxes.unbind(1)
        return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

    def _prepare_for_coco_detection(self, predictions):
        coco_results = []
        for image_id, prediction in predictions.items():
            boxes = prediction["boxes"]
            boxes =  self._convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
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
    

    # Metrics

    def _summarize_metrics(self, evaluator):
        metrics_buffer = StringIO()
        sys.stdout = metrics_buffer
        evaluator.summarize()
        sys.stdout = STDOUT
        
        metrics = metrics_buffer.getvalue()
        metrics = metrics.split('\n')
        metrics = [m for m in metrics if 'Average' in m]
        metrics_dict = {}
        for metric in metrics:
            name, value = metric.split(' = ')
            metrics_dict[name[1:]] = float(value)
        return metrics_dict


    def evaluate(self, valid_dataset, valid_loader, threshold, save_plots=False):
        print(Fore.MAGENTA, "Evaluating on threshold: ", threshold, Fore.WHITE)
        evaluator = CocoEvaluator(
            coco_gt=valid_dataset.coco, 
            iou_types=["bbox"]
        )
        # Evaluate Coco Predictions
        empty_predictions = True
        self.model.eval()
        for batch_id, batch in enumerate(valid_loader):
            
            image_ids = [label['image_id'] for label in batch['labels']]
            predictions = self.generate_predictions(batch, image_ids, threshold)
            
            # Plots
            if save_plots and (batch_id == 0 or batch_id == len(valid_loader) - 1):
                self.plotter.plot_batch_comparison(predictions, valid_dataset, threshold, batch_id)
            
            # predictions = self._prepare_for_coco_detection(predictions)
            predictions = []
            if len(predictions) != 0:
                empty_predictions = False
                evaluator.update(predictions)
            
        if empty_predictions:
            metrics_dict = {metric: 0.0 for metric in metrics_names}
            return metrics_dict
            
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
    
        # Metrics
        
        metrics = self._summarize_metrics(evaluator)
        return metrics


