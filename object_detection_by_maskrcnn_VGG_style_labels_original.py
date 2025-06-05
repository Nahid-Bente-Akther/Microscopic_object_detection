# https://youtu.be/QntADriNHuk
"""
Mask R-CNN - Multiclass - VGG style annotations in JSON format

For annotations, use one of the following programs: 
    https://www.makesense.ai/

"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes

from mrcnn.utils import Dataset
from matplotlib import pyplot as plt

from mrcnn.config import Config
from mrcnn.model import MaskRCNN


from mrcnn import model as modellib, utils

###############################################################################

#Define class for label, annotation and mask for the annotated custom dataset

###############################################################################

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        """Load a subset of the custom dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes according to the numbe of classes required to detect
        self.add_class("custom", 1, "microcoleus")
        self.add_class("custom",2,"no_microcoleus")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "labels/labels_demo_VGG_like.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            #labelling each class in the given image to a number

            custom = [s['region_attributes'] for s in a['regions'].values()]
            
            num_ids=[]
            #Add the classes according to the requirement
            for n in custom:
                try:
                    if n['label']=='microcoleus':
                        num_ids.append(1)
                    elif n['label']=='no_microcoleus':
                        num_ids.append(2)
                except:
                    pass

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "custom",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a custom dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "custom":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = image_info['num_ids']	
        #print("Here is the numID",num_ids)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        num_ids = np.array(num_ids, dtype=np.int32)	
        return mask, num_ids#.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32), 

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "custom":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

###############################################################################

#Load custom dataset

###############################################################################

#                           *******Training data**********

###############################################################################


dataset_train = CustomDataset()
#dataset_train.load_custom("demo_img/", "train") 
dataset_train.load_custom("C:/Users/nahid/Downloads/Mask-RCNN-TF2/demo_img/", "train") 
dataset_train.prepare()
print('Training data : %d' % len(dataset_train.image_ids))


###############################################################################

#Check how many bacteria are there of both classes inside this training dataset

###############################################################################

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

###############################################################################

#                       ********Validation data**********

###############################################################################

#How many image data are there in validatin folder
dataset_val = CustomDataset()
#dataset_val.load_custom("demo_img/", "val")
dataset_val.load_custom("C:/Users/nahid/Downloads/Mask-RCNN-TF2/demo_img/", "val")  
dataset_val.prepare()
print('Validation data : %d' % len(dataset_val.image_ids))

###############################################################################

#Check how many bacteria are there of both class inside this validation dataset

###############################################################################

# Load the annotated data

with open("C:\\Users\\nahid\Downloads\\Mask-RCNN-TF2\\demo_img\\val\\labels\\labels_demo_VGG_like.json") as f:
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

# Load annotation

# define image id
image_id = 15
# load the image
image = dataset_train.load_image(image_id)
# load the masks and the class ids
mask, class_ids = dataset_train.load_mask(image_id)
#print('mask : ', mask , 'class_ID : ', class_ids)###############


# display_instances(image, r1['rois'], r1['masks'], r1['class_ids'],
# dataset.class_names, r1['scores'], ax=ax, title="Predictions1")

# extract bounding boxes from the masks
bbox = extract_bboxes(mask)
# display image with masks and bounding boxes
display_instances(image, bbox, mask, class_ids, dataset_train.class_names)


# define a configuration for the model
class MarbleConfig(Config):
    # define the name of the configuration
    NAME = "micro_cfg"
    # number of classes (background + microcoleus + no_microcoleus)
    NUM_CLASSES = 1 + 2
    # number of training steps per epoch
    STEPS_PER_EPOCH = 50
    # Set RPN anchor scales
    #RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)  # Modify this line by me
    DETECTION_MIN_CONFIDENCE = 0.9 # Skip detections with < 90% confidence 
# prepare config
config = MarbleConfig()
config.display() 




ROOT_DIR = os.path.abspath("C:\\Users\\nahid\\Downloads\\Mask-RCNN-TF2")
print(ROOT_DIR)
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
# Directory to save logs and trained model
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
print(DEFAULT_LOGS_DIR)
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "coco_weights\mask_rcnn_coco.h5")
print(COCO_WEIGHTS_PATH)

########################
#Weights are saved to root D: directory. need to investigate how they can be
#saved to the directory defined... "logs_models"

###############################################################################

# define the model

###############################################################################

model = MaskRCNN(mode='training', model_dir=DEFAULT_LOGS_DIR, config=config)
print(DEFAULT_LOGS_DIR)
# load weights (mscoco) and exclude the output layers
print(COCO_WEIGHTS_PATH)
model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
print(config.LEARNING_RATE)
# train weights (output layers or 'heads')
#model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE / 10, epochs=10, layers='all')
#model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE / 10, epochs=10, layers='4+')
model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=100, layers='heads')

###############################################################################

#validation and Loss curve visualization 

###############################################################################

#On the anaconda prompt
#tensorboard --logdir=C:/Users/nahid/Downloads/Mask-RCNN-TF2/logs/

###############################################################################

#INFERENCE

###############################################################################

from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from mrcnn.utils import compute_ap
from numpy import expand_dims
from numpy import mean
from matplotlib.patches import Rectangle


# define the prediction configuration
class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "micro_cfg"
	# number of classes (background + Blue Marbles + Non Blue marbles)
	NUM_CLASSES = 1 + 2
	# Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
 
# calculate the mAP for a model on a given dataset
def evaluate_model(dataset, model, cfg):
	APs = list()
	for image_id in dataset.image_ids:
		# load image, bounding boxes and masks for the image id
		image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = expand_dims(scaled_image, 0)
		# make prediction
		yhat = model.detect(sample, verbose=0)
		# extract results for first sample
		r = yhat[0]
		# calculate statistics, including AP
		AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
		# store
		APs.append(AP)
	# calculate the mean AP across all images
	mAP = mean(APs)
	return mAP

    
###############################################################################

# create config

###############################################################################

cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='logs', config=cfg)
# load model weights
model.load_weights('C:\\Users\\nahid\\Downloads\\Mask-RCNN-TF2\\logs\\mask_rcnn_micro_cfg_0022.h5', by_name=True)

# evaluate model on training dataset
train_mAP = evaluate_model(dataset_train, model, cfg)
print("Train mAP: %.3f" % train_mAP)

# evaluate model on test dataset
test_mAP = evaluate_model(dataset_val, model, cfg)
print("Test mAP: %.3f" % test_mAP)

###############################################################################

#Test saved model on a single image which is completely unseen by the model

###############################################################################
#C:/Users/nahid/Downloads/Mask-RCNN-TF2/demo_img/test
#marbles_img = skimage.io.imread("demo_img/010.jpg")

#marbles_img = skimage.io.imread("demo_img/014.jpg")

test_img = skimage.io.imread("C:/Users/nahid/Downloads/Mask-RCNN-TF2/demo_img/test/028.jpg")

#marbles_img = skimage.io.imread("demo_img/030.jpg")
plt.imshow(test_img)

detected = model.detect([test_img])
results = detected[0]
class_names = ['BG', 'microcoleus', 'no_microcoleus']
display_instances(test_img, results['rois'], results['masks'], 
                  results['class_ids'], class_names, results['scores'])




###############################################################################

#Evaluation : Confusion Matrix

###############################################################################

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt

y_true = []
y_pred = []

for image_id in dataset_val.image_ids:
    # Load image and GT
    image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset_val, cfg, image_id, use_mini_mask=False)
    molded_image = mold_image(image, cfg)
    sample = expand_dims(molded_image, 0)

    # Predict
    result = model.detect(sample, verbose=0)[0]

    # For simplicity, compare only 1-to-1 match: most confident prediction
    if len(gt_class_id) > 0 and len(result['class_ids']) > 0:
        y_true.append(gt_class_id[0])  # take first GT label
        y_pred.append(result['class_ids'][0])  # take first predicted label

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[1, 2])  # 1: microcoleus, 2: no_microcoleus

# Display
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['microcoleus', 'no_microcoleus'],
            yticklabels=['microcoleus', 'no_microcoleus'], cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

###############################################################################

#precision, recall, F1 score

###############################################################################

from sklearn.metrics import precision_score, recall_score, f1_score
from mrcnn.utils import compute_ap
import numpy as np

# Initialize accumulators
true_labels = []
pred_labels = []

for image_id in dataset_val.image_ids:
    # Load GT data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset_val, cfg, image_id, use_mini_mask=False)
    molded_image = mold_image(image, cfg)
    sample = np.expand_dims(molded_image, 0)
    
    # Predict
    result = model.detect(sample, verbose=0)[0]

    # Match predicted to GT via IoU
    AP, precisions, recalls, overlaps = compute_ap(gt_bbox, gt_class_id, gt_mask,
                                                   result['rois'], result['class_ids'],
                                                   result['scores'], result['masks'])

    # Accumulate matched class IDs
    gt_len = len(gt_class_id)
    pred_len = len(result['class_ids'])

    min_len = min(gt_len, pred_len)

    # Add matched IDs (limited by min count to avoid mismatch)
    true_labels.extend(gt_class_id[:min_len])
    pred_labels.extend(result['class_ids'][:min_len])

# Now compute metrics
print("Precision:", precision_score(true_labels, pred_labels, average='weighted'))
print("Recall:", recall_score(true_labels, pred_labels, average='weighted'))
print("F1 Score:", f1_score(true_labels, pred_labels, average='weighted'))

# Optional detailed report
from sklearn.metrics import classification_report
print("\nDetailed Report:")
print(classification_report(true_labels, pred_labels, target_names=['microcoleus', 'no_microcoleus']))

##################################################

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from mrcnn.utils import compute_ap, compute_matches
import numpy as np

true_labels = []
pred_labels = []

for image_id in dataset_val.image_ids:
    # Load ground truth
    image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(
        dataset_val, cfg, image_id, use_mini_mask=False
    )
    molded_image = mold_image(image, cfg)
    sample = np.expand_dims(molded_image, 0)

    # Run detection
    result = model.detect(sample, verbose=0)[0]

    # Match predictions to ground truth using IoU
    overlaps, matched_gt, matched_pred = compute_matches(
        gt_bbox, gt_class_id, gt_mask,
        result['rois'], result['class_ids'], result['scores'], result['masks'],
        iou_threshold=0.5  # You can adjust this threshold
    )

    # -------------------------------
    # Add matched pairs
    for gt_idx, pred_idx in zip(matched_gt, matched_pred):
        true_labels.append(gt_class_id[gt_idx])
        pred_labels.append(result['class_ids'][pred_idx])

    # -------------------------------
    # Add unmatched ground truths as FN (missed detections)
    unmatched_gt_indices = list(set(range(len(gt_class_id))) - set(matched_gt))
    for gt_idx in unmatched_gt_indices:
        true_labels.append(gt_class_id[gt_idx])
        pred_labels.append(-1)  # Label -1 for "not detected"

    # -------------------------------
    # Add unmatched predictions as FP (false detections)
    unmatched_pred_indices = list(set(range(len(result['class_ids']))) - set(matched_pred))
    for pred_idx in unmatched_pred_indices:
        true_labels.append(-1)  # Ground truth "nothing"
        pred_labels.append(result['class_ids'][pred_idx])

# -------------------------------
# Filter out unknown class (-1) if you want
label_names = ['microcoleus', 'no_microcoleus']

print("\n📊 Classification Report:")
print(classification_report(true_labels, pred_labels, target_names=label_names, labels=[1, 0]))  # 1: microcoleus, 0: no_microcoleus

print("✅ Precision:", precision_score(true_labels, pred_labels, average='weighted', labels=[1, 0]))
print("✅ Recall:", recall_score(true_labels, pred_labels, average='weighted', labels=[1, 0]))
print("✅ F1 Score:", f1_score(true_labels, pred_labels, average='weighted', labels=[1, 0]))

###############################################################################

Viusaliztion test

###############################################################################

import matplotlib.pyplot as plt
import numpy as np

# Example image for testing
image = np.random.rand(100, 100, 3)  # Random image for testing purposes

# Function to display image and show title
def show_image_with_metrics(image, title):
    plt.imshow(image)
    plt.title(title)
    plt.show()  # This should display the image in a window

# Test the plotting function
show_image_with_metrics(image, "Test Image")


###############################################################################

#Show detected objects in color and all others in B&W    

###############################################################################

'''
def color_splash(img, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(img)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, img, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash

import skimage
def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        #print("Running on {}".format(img))
        # Read image
        img = skimage.io.imread(image_path)
        # Detect objects
        r = model.detect([img], verbose=1)[0]
        # Color splash
        splash = color_splash(img, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, img = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                img = img[..., ::-1]
                # Detect objects
                r = model.detect([img], verbose=0)[0]
                # Color splash
                splash = color_splash(img, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)

detect_and_color_splash(model, image_path="demo_img/val/053.jpg")'''

###############################################################################
                         
