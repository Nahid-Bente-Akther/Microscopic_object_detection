# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 07:06:36 2024

@author: nahid
"""

import json

# Load the annotations
with open("C:\\Users\\nahid\Downloads\\Mask-RCNN-TF2\\demo_img\\train\\labels\\labels_demo_VGG_like.json") as f:
    annotations = json.load(f)

class_counts = {"microcoleus": 0, "no_microcoleus": 0}

# Loop through the annotations and count classes
for annotation in annotations.values():
    for region in annotation['regions'].values():
        class_label = region['region_attributes']['label']
        if class_label == 'microcoleus':
            class_counts["microcoleus"] += 1
        elif class_label == 'no_microcoleus':
            class_counts["no_microcoleus"] += 1

print("Class distribution:", class_counts)
