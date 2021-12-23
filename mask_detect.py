import detectron2
from detectron2.utils.logger import setup_logger
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

classes = ['0','1','2']

# Config
config_file = 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(config_file))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75 # Threshold
cfg.MODEL.WEIGHTS = "model/model_final.pth" # path to the model we just trained
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
cfg.MODEL.DEVICE = 'cpu'

# Create predictor
predictor = DefaultPredictor(cfg)

# Register data
# DatasetCatalog.register('pub_test', lambda: get_mask_dicts('D:/tailieu/triTueNhanTao/CV/mask/dataset/images/public_test'))
mask_metadata = MetadataCatalog.get('pub_test').set(thing_classes=classes)

# Predict on image
def predict_detectron(img):
    im = img
    outputs = predictor(im)
    result = []
    for box, cls in zip(outputs["instances"].pred_boxes.to('cpu'), outputs["instances"].pred_classes.to('cpu')):
        textCls = str(cls)
        temp = [[int(box[0]), int(box[1]), int(box[2]), int(box[3])], textCls]
        result.append(temp)
    # print(outputs)
    # temp = []

    # for box, cls in zip(outputs["instances"].pred_boxes.to('cpu'), outputs["instances"].pred_classes.to('cpu')):
    #     cv2.rectangle(img, (int(temp[0]), int(temp[1])), (int(temp[2]), int(temp[3])), color=(0,255,0), thickness=2)
    #     textCls = str(cls)
    #     cv2.putText(img, textCls, thickness=2)

    # print(type(temp[0]))
    # v = Visualizer(im[:, :, ::-1],
    #             metadata=mask_metadata,
    #             scale=0.7,
    #             instance_mode=ColorMode.IMAGE
    #             )
    # im = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2.imshow('img', out.get_image()[:, :, ::-1])
    # cv2.rectangle(im, (int(temp[0]), int(temp[1])), (int(temp[2]), int(temp[3])), color=(0,255,0), thickness=2)

    return result

# vcap = cv2.VideoCapture(0)
#
# while (1):
#     ret, frame = vcap.read()
#     # print(frame)
#     predict_detectron(frame)
#     cv2.imshow('video', frame)
#     if (cv2.waitKey(1) & 0xFF == ord('q')):
#         break