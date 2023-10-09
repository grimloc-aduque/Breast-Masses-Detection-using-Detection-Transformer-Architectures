
import sys
from io import StringIO

import torch
from coco_eval import CocoEvaluator

from detr_config import Config
from detr_factory import DETRFactory
from detr_metrics import metrics_names
from detr_plotter import Plotter

STDOUT = sys.stdout

class ModelEvaluator:

    def __init__(self, model, detr_factory:DETRFactory, plotter:Plotter):
        self.model = model
        self.image_processor = detr_factory.new_image_processor()
        self.plotter = plotter

    # Coco Formating

    def generate_predictions(self, batch, threshold):
        pixel_values = batch["pixel_values"].to(Config.DEVICE)
        pixel_mask = batch["pixel_mask"].to(Config.DEVICE)
        labels = [{k: v.to(Config.DEVICE) for k, v in t.items()} for t in batch["labels"]] 
        with torch.no_grad():
            outputs = self.model(
                pixel_values = pixel_values,
                pixel_mask = pixel_mask
            )
        orig_target_sizes = torch.stack([label["orig_size"] for label in labels], dim=0)        
        results = self.image_processor.post_process_object_detection(
            outputs, target_sizes=orig_target_sizes, threshold=threshold)
        predictions = {label['image_id'].item():output for label, output in zip(labels, results)}
        return predictions # bbox: (x_min, y_min, x_max, y_max)

    def _convert_to_xywh(self, boxes):
        xmin, ymin, xmax, ymax = boxes.unbind(1)
        return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

    def _prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue
            boxes = prediction["boxes"]
            boxes =  self._convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            coco_results.extend(
                [
                    {
                        "image_id": original_id,
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


    def evaluate(self, valid_dataset, valid_loader, threshold):
        print("Evaluating on threshold: ", threshold)
        evaluator = CocoEvaluator(
            coco_gt=valid_dataset.coco, 
            iou_types=["bbox"]
        )
        # Evaluate Coco Predictions
        empty_predictions = True
        self.model.eval()
        for batch_id, batch in enumerate(valid_loader):
            predictions = self.generate_predictions(batch, threshold)
            
            # Plots
            self.plotter.plot_batch_comparison(predictions, valid_dataset, threshold, batch_id)
            
            predictions = self._prepare_for_coco_detection(predictions)
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


