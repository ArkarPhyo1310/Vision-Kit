import contextlib
import io
import json
import tempfile
from collections import defaultdict

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate
from yolo_series.utils.bboxes import xyxy_to_xywh
from yolo_series.utils.postprocess import scale_coords


class COCOEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self,
        class_ids: list,
        gt_json: str = "D:/Datasets/coco/coco128/train.json",
        img_size: int = (640, 640),
        per_class_AP: bool = False,
        per_class_AR: bool = False
    ) -> None:
        self.img_size = img_size
        self.per_class_AP = per_class_AP
        self.per_class_AR = per_class_AR
        self.class_id = class_ids
        self.gt_json = gt_json

    def convert_to_coco(self, outputs, img_infos, ids):
        data_list = []
        image_wise_data = defaultdict(dict)
        for (output, img_info, img_id) in zip(outputs, img_infos, ids):
            if output is None:
                continue

            output = scale_coords(self.img_size, output, img_info)
            output = output.cpu()
            bboxes = output[:, 0:4]
            scores = output[:, 4]
            cls = output[:, 5]

            image_wise_data.update({
                int(img_id): {
                    "bboxes": [box.numpy().tolist() for box in bboxes],
                    "scores": [score.numpy().item() for score in scores],
                    "categories": [
                        self.class_id[int(cls[ind])]
                        for ind in range(bboxes.shape[0])
                    ],
                }
            })

            bboxes = xyxy_to_xywh(bboxes)

            for ind in range(bboxes.shape[0]):
                label = self.class_id[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)

        return data_list, image_wise_data

    def evaluate_prediction(self, data_dict):
        print("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]
        info = ""

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = COCO(self.gt_json)
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)

            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info = redirect_string.getvalue()
            cat_ids = list(cocoGt.cats.keys())
            cat_names = [cocoGt.cats[catId]['name']
                         for catId in sorted(cat_ids)]
            # if self.per_class_AP:
            #     AP_table = per_class_AP_table(cocoEval, class_names=cat_names)
            #     info += "per class AP:\n" + AP_table + "\n"
            # if self.per_class_AR:
            #     AR_table = per_class_AR_table(cocoEval, class_names=cat_names)
            #     info += "per class AR:\n" + AR_table + "\n"
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info
