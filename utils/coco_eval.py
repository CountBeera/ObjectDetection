import numpy as np

# Patch numpy for cocoeval compatibility
if not hasattr(np, "float"):
    np.float = float

from pycocotools.cocoeval import Params, COCOeval
import utils.coco_utils as coco_utils

# Fully override Params.setDetParams
def _patched_setDetParams(self):
    thr_min, thr_max, step = 0.5, 0.95, 0.05
    num_iou = int(np.round((thr_max - thr_min) / step)) + 1
    self.iouThrs = np.linspace(thr_min, thr_max, num_iou, endpoint=True)
    self.recThrs = np.linspace(0.0, 1.0, 101, endpoint=True)
    self.maxDets = [1, 10, 100]
    self.areaRng = [
        [0**2, 1e5**2],
        [0**2, 32**2],
        [32**2, 96**2],
        [96**2, 1e5**2],
    ]
    self.areaRngLbl = ['all', 'small', 'medium', 'large']
    self.useCats = 1
    self.catIds = []
    self.imgIds = []

Params.setDetParams = _patched_setDetParams


class CocoEvaluator:
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        # Instantiate one COCOeval per IoU type
        self.coco_eval = {t: COCOeval(coco_gt, iouType=t) for t in iou_types}
        self.img_ids = []

    def update(self, predictions):
        # Normalize to list of dicts
        if isinstance(predictions, dict):
            preds = []
            for img_id, out in predictions.items():
                p = out.copy()
                p["image_id"] = img_id
                preds.append(p)
        else:
            preds = predictions

        # Collect image IDs
        img_ids = list(np.unique([p["image_id"] for p in preds]))
        self.img_ids.extend(img_ids)

        # Feed detections into each evaluator
        for iou_type, coco_eval in self.coco_eval.items():
            results = coco_utils.prepare_for_coco_detection(preds)
            if not results:
                continue  # no detections at all, skip
            coco_dt = self.coco_gt.loadRes(results)
            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = img_ids
            coco_eval.evaluate()

    def synchronize_between_processes(self):
        pass

    def accumulate(self):
        for ev in self.coco_eval.values():
            ev.accumulate()

    # def summarize(self):
    #     for ev in self.coco_eval.values():
    #         ev.summarize()
    def summarize(self):
        metrics = {}
        for iou_type, ev in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")
            ev.summarize()

            # Extract specific metrics
            stats = ev.stats
            metrics[iou_type] = {
                "AP@[IoU=0.50:0.95]": stats[0],
                "AP@50": stats[1],
                "AP@75": stats[2],
                "AR@1": stats[6],
                "AR@10": stats[7],
                "AR@100": stats[8],
            }

            print(f"→ Precision (AP@50): {stats[1]:.3f}")
            print(f"→ Recall    (AR@100): {stats[8]:.3f}")

        return metrics

