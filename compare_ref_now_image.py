import numpy as np

import cv2
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from predict import Predictor
from torch import Tensor

predictor = Predictor()

im_dirty = cv2.imread("./custom_images/dirty.jpg")
im_clean = cv2.imread("./custom_images/clean.jpg")

z_dirty, m_dirty = predictor.predict('./custom_images/dirty.jpg', vocabulary="lvis", custom_vocabulary=None, save=False)
z_clean, m_clean = predictor.predict('./custom_images/clean.jpg', vocabulary="lvis", custom_vocabulary=None, save=False)

instances_dirty: Instances = z_dirty["instances"]
instances_clean: Instances = z_clean["instances"]

# MANUALS
# NOTE how to check the fileds
# instances_dirty.get_fields().keys()
# NOTE how to get classes from metadata
# m_dirty.thing_classes
# NOTE how to check the class of a specific instance
# IN: instances_dirty.get_fields()["pred_classes"]
# OUT: tensor([ 231,  231,  420,  880,  686,  231,  295,  231,  126,  654,  389,  630,
#   142,  697,  184,  142,  328,  389,  142,  630,  389,  389, 1140,  142,
#   360, 1071,  350,  180,  343, 1094, 1005, 1119,  142, 1071,  366, 1170,
#  1154,  128,  836, 1177,  897,  343, 1085,  961,  138, 1177],
# device='cuda:0')
# IN: m_dirty.thing_classes[1177]
# OUT: 'wheel'

# get index of dining table
idx_dining_table = m_clean.thing_classes.index("dining_table")
field_of_dining_table = instances_clean.get_fields()["pred_classes"] == idx_dining_table # Tensor of bools

# get mask of dining table
bin_mask_of_dining_table = instances_clean.get_fields()["pred_masks"][field_of_dining_table][0].cpu().numpy()
mask_of_dining_table = np.where(bin_mask_of_dining_table, 255, 0).astype(np.uint8)
rgb_mask_of_dining_table = cv2.cvtColor(mask_of_dining_table, cv2.COLOR_GRAY2RGB)

# get box of dining table
box_of_dining_table = instances_clean.get_fields()["pred_boxes"][field_of_dining_table].tensor.cpu().numpy().astype(np.int32).squeeze()

# get masked image of dining table
im_clean_dining_table_masked = cv2.bitwise_and(im_clean, rgb_mask_of_dining_table)
im_dirty_dining_table_masked = cv2.bitwise_and(im_dirty, rgb_mask_of_dining_table)

# get cropped image of dining table. 2 images are same size
im_clean_dining_table_cropped = im_clean[box_of_dining_table[1]:box_of_dining_table[3], box_of_dining_table[0]:box_of_dining_table[2]]
im_dirty_dining_table_cropped = im_dirty[box_of_dining_table[1]:box_of_dining_table[3], box_of_dining_table[0]:box_of_dining_table[2]]


###
z_clean_dining_table, m_clean_dining_table = predictor.predict(im_clean_dining_table_cropped, vocabulary="lvis", custom_vocabulary=None, save=False)
z_dirty_dining_table, m_dirty_dining_table = predictor.predict(im_dirty_dining_table_cropped, vocabulary="lvis", custom_vocabulary=None, save=False)

instances_clean_dining_table: Instances = z_clean_dining_table["instances"]
instances_dirty_dining_table: Instances = z_dirty_dining_table["instances"]

# get index of dining table
idx_dining_table = m_clean_dining_table.thing_classes.index("dining_table")
field_of_dining_table = instances_clean_dining_table.get_fields()["pred_classes"] == idx_dining_table # Tensor of bools


### Do these processes for all classes
for cls_index in instances_dirty_dining_table.get_fields()['pred_classes']:
    field_of_object = instances_dirty_dining_table.get_fields()["pred_classes"] == cls_index
    bin_mask = instances_dirty_dining_table.get_fields()["pred_masks"][field_of_object][0].cpu().numpy()
    mask = np.where(bin_mask, 255, 0).astype(np.uint8)
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    im_dirty_dining_table_obj_masked = cv2.bitwise_and(im_dirty_dining_table_cropped, rgb_mask)
