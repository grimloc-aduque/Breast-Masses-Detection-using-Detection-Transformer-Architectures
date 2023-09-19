
import torch

def convert_to_xywh(boxes):
    center_x, center_y, width, height = boxes.unbind(1)
    boxes = torch.stack((center_x, center_y, width, height), dim=1)
    boxes = boxes.tolist()
    return boxes


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