import os
import sys
import math
import random
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import json

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

sys.path.append(os.path.join("samples/coco/"))
import coco

MODEL_DIR = os.path.join("logs")
COCO_MODEL_PATH = os.path.join("mask_rcnn_coco.h5")

IMAGE_DIR = os.path.join("test_images")


class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                          config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

file_names = next(os.walk(IMAGE_DIR))[2]
for file in file_names:
    image = skimage.io.imread(os.path.join(IMAGE_DIR, file))
    results = model.detect([image], verbose=1)

    r = results[0]

    # visualize.display_instances(image, r['rois'], r['masks'],
    #                             r['class_ids'],
    #                             class_names, r['scores'])

    r['rois'] = r['rois'].tolist()
    r['class_ids'] = r['class_ids'].tolist()
    r['scores'] = r['scores'].tolist()
    r['masks'] = r['masks'].tolist()
    # del r['masks']

    fname = file.split('.')[0] + '.json'
    json.dump(r, open('output/' + fname, 'w'), indent=2)

# 1,3 lie; 2,4 hang
