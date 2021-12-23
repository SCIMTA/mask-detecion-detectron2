import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

setup_logger()
from datetime import datetime

# import some common libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode
import glob
import os
import sys
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from detectron2.data import DatasetCatalog, MetadataCatalog

classes = ['0', '1', '2']

# Config
config_file = 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(config_file))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # Threshold
cfg.MODEL.WEIGHTS = "model/model_final.pth"  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set a custom testing threshold
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
cfg.MODEL.DEVICE = 'cuda'   # cpu or cuda

# Create predictor
predictor = DefaultPredictor(cfg)

# Register data
# DatasetCatalog.register('pub_test', lambda: get_mask_dicts('D:/tailieu/triTueNhanTao/CV/mask/dataset/images/public_test'))
mask_metadata = MetadataCatalog.get('pub_test').set(thing_classes=classes)


# Predict on image
def predict_detectron(img):
    """
        Predict function
        input:
            img: image read from opencv
        output:
            [[x,y,w,h], class]
    """
    im = img
    outputs = predictor(im)
    result = []

    for box, cls in zip(outputs["instances"].pred_boxes.to('cpu'), outputs["instances"].pred_classes.to('cpu')):
        color = (0, 0, 0)
        textCls = "unknown"
        if int(cls) == 0:
            color = (255, 0, 0)
            textCls = 'without_mask'
        if int(cls) == 1:
            color = (0, 255, 0)
            textCls = 'with_mask'
        if int(cls) == 2:
            color = (0, 0, 255)
            textCls = 'incorrect_mask'

        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=color, thickness=2)
        temp = [[int(box[0]), int(box[1]), int(box[2]), int(box[3])], textCls]
        result.append(temp)
        cv2.putText(img, textCls, org=(int(box[0]), int(box[1] - 10)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    thickness=1, color=color)

    return result
