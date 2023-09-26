

import sys
from io import StringIO

import torch
from coco_eval import CocoEvaluator
from detr_dataset import get_dataloader
from detr_config import Config

STDOUT = sys.stdout


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes =  convert_to_xywh(boxes).tolist()
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


def get_metrics(model, dataset, image_processor, threshold):
    evaluator = CocoEvaluator(
        coco_gt=dataset.coco, 
        iou_types=["bbox"]
    )
    
    dataloader = get_dataloader(dataset, image_processor)
    
    valid_predictions = False
    model.eval()
    for batch in dataloader:
        
        pixel_values = batch["pixel_values"].to(Config.DEVICE)
        pixel_mask = batch["pixel_mask"].to(Config.DEVICE)
        labels = [{k: v.to(Config.DEVICE) for k, v in t.items()} for t in batch["labels"]] 
        
        with torch.no_grad():
            outputs = model(
                pixel_values = pixel_values,
                pixel_mask = pixel_mask
            )
        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)        
        results = image_processor.post_process_object_detection(
            outputs, target_sizes=orig_target_sizes, threshold=threshold)
        
        predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}
        predictions = prepare_for_coco_detection(predictions)
        
        if len(predictions) != 0:
            valid_predictions = True
            evaluator.update(predictions)
        
    if valid_predictions:
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
    
        # Metrics
        
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
            
    else:
        metrics_dict = {
            'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]': 0.0,
            'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]': 0.0,
            'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]': 0.0,
            'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]': 0.0,
            'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]': 0.0,
            'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]': 0.0,
            'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]': 0.0,
            'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]': 0.0,
            'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]': 0.0,
            'Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]': 0.0,
            'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]': 0.0,
            'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]': 0.0
        }
        
    return metrics_dict