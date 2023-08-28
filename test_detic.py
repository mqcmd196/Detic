from predict import Predictor

if __name__ == '__main__':
    predictor = Predictor()
    print("output: {}".format(predictor.predict('./custom_images/dirty.jpg', vocabulary="lvis", custom_vocabulary=None)))
