import cv2
from predict import Predictor

predictor = Predictor()

im_dirty = cv2.imread("./custom_images/dirty.jpg")
im_clean = cv2.imread("./custom_images/clean.jpg")

z_dirty = predictor.predict('./custom_images/dirty.jpg', vocabulary="lvis", custom_vocabulary=None, save=False)
z_clean = predictor.predict('./custom_images/clean.jpg', vocabulary="lvis", custom_vocabulary=None, save=False)

instances_dirty = z_dirty["instances"]
instances_clean = z_clean["instances"]
