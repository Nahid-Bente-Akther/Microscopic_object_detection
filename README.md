# Cyanobacteria_detection

The Microscopic Object Detection project edits the original Mask_RCNN project and Mask_RCNN_TF2, which only supports TensorFlow 1.0 and TensorFlow 2.0 respectively. For this project, the Mask R-CNN can be trained and tested (i.e make predictions) in Tensorflow 2.2.0 is used. The requirement.txt file contains all of the packages with their version for this work.

To use this model, it is necessary to set the environment according to the version and annotate custom data properly.

How to use the .py file :

1. First create a work directory and clone the Mask_RCNN_TF2 model inside the directory.
2. Download the pretrained weight of the Mask R-CNN model based on the COCO dataset. The trained weights can be downloaded from this link: https://github.com/ahmedfgad/Mask-RCNN-TF2/releases/download/v3.0/mask_rcnn_coco.h5 and then store in the coco_weight folder in the working directory.
3. Keep all custom image data inside the working directory.
4. Keep the requirement.txt file inside  the woking directory.
5. After set the working directory and environment, run the .py script.

The link https://github.com/ahmedfgad/Mask-RCNN-TF2 of pre-trained Mask_RCNN_TF2 version 2.0 

