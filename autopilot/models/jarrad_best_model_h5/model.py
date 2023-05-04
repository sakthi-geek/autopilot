from tensorflow import keras
import numpy as np
import imutils
import cv2
import os
import tensorflow as tf


class Model:

    saved_model = 'best-angle-model.h5'

    def __init__(self):
        self.model = keras.models.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.saved_model))
        self.model.summary()

    def preprocess(self, image):
        import tensorflow as tf
        im = tf.image.resize(image, (240, 320))
        im = tf.divide(im, 255)  # Normalize, need to check if needed
        # im = tf.expand_dims(im, axis=0)  # add batch dimension
        # cv2.imshow('im', im)
        print(im.shape)
        return im

    def predict(self, image):
        image = self.preprocess(image)
        angle = self.model.predict(np.array([image]))[0]
        # Training data was normalised so convert back to car units
        angle = 80 * np.clip(angle, 0, 1) + 50
        # speed = 35 * np.clip(speed, 0, 1)
        return angle#, speed
