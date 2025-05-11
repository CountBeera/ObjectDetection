def prepare_for_coco_detection(predictions):
    coco_results = []

    for prediction in predictions:
        boxes = prediction["boxes"]
        scores = prediction["scores"]
        labels = prediction["labels"]
        image_id = prediction["image_id"]

        boxes = boxes.tolist()
        scores = scores.tolist()
        labels = labels.tolist()

        for k in range(len(boxes)):
            box = boxes[k]
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            coco_results.append({
                "image_id": image_id,
                "category_id": labels[k],
                "bbox": [x_min, y_min, width, height],
                "score": scores[k],
            })

    return coco_results
