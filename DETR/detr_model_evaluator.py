
import sys
from io import StringIO

import torch
from coco_eval import CocoEvaluator
from detr_config import Config
from detr_metrics import metrics_names
import copy

STDOUT = sys.stdout

class ModelEvaluator:

    def __init__(self, model, detr_factory):
        self.model = model
        self.image_processor = detr_factory.new_image_processor()

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
        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)        
        results = self.image_processor.post_process_object_detection(
            outputs, target_sizes=orig_target_sizes, threshold=threshold)
        predictions = {target['image_id'].item():output for target, output in zip(labels, results)}
        return predictions

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
        are_predictions = False
        self.model.eval()
        for batch in valid_loader:
            predictions = self.generate_predictions(batch, threshold)
            predictions = self._prepare_for_coco_detection(predictions)
            if len(predictions) != 0:
                are_predictions = True
                evaluator.update(predictions)
            
        if not are_predictions:
            metrics_dict = {metric: 0.0 for metric in metrics_names}
            return metrics_dict
            
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
    
        # Metrics
        
        metrics = self._summarize_metrics(evaluator)
        return metrics


