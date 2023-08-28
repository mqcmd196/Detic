import cv2
from detectron2.structures.instances import Instances
from predict import Predictor

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
