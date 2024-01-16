import tensorflow as tf
import numpy as np
import os
import cv2
import keras
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

def resizeImages(input_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image

initializer = tf.random_normal_initializer(0., 0.02)

class Model(tf.keras.Model):
    def __init__(self, classes):
        super(Model, self).__init__()
        self.c1_1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1),kernel_regularizer=tf.keras.regularizers.l2(0.01),padding='valid')
        self.c1_2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='valid')
        self.c2_1 = tf.keras.layers.Conv2D(64,(3, 3),activation='relu', padding='valid')
        self.c2_2 = tf.keras.layers.Conv2D(64,(3, 3),activation='relu', padding='valid')
        self.c3_1 = tf.keras.layers.Conv2D(128,(3, 3),activation='relu', padding='valid')
        self.c3_2 = tf.keras.layers.Conv2D(128,(3, 3),activation='relu', padding='valid')
        self.c4_1 = tf.keras.layers.Conv2D(256,(3, 3),activation='relu', padding='valid')
        self.c4_2 = tf.keras.layers.Conv2D(256,(3, 3),activation='relu', padding='valid')
        self.m1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.m2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.m3 = tf.keras.layers.MaxPooling2D((2, 2))
        self.m4 = tf.keras.layers.MaxPooling2D((2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu',  kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.classifier = tf.keras.layers.Dense(classes, activation='softmax')

    def call_model(self):
        inputs = tf.keras.layers.Input(shape = [256, 256, 1])
        x = inputs
        x = self.c1_1(inputs)
        x = self.c1_2(x)
        x = self.m1(x)
        x = self.c2_1(x)
        x = self.c2_2(x)
        x = self.m2(x)
        x = self.c3_1(x)
        x = self.c3_2(x)
        x = self.m3(x)
        x = self.c4_1(x)
        x = self.c4_2(x)
        x = self.m4(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.classifier(x)
        
        return tf.keras.Model(inputs = inputs, outputs=x)

        
model = Model(classes=6).call_model()
model.summary()

model.load_weights("model.h5")

class_names = ["01_palm","02_l","05_thumb","06_index","07_ok","09_c"]
class_labels = {0:"01_palm", 1:"02_l", 2:"05_thumb", 3:"06_index", 4:"07_ok", 5:"09_c"}



cap = cv2.VideoCapture(0)

while True:
    ret, img1 = cap.read()
    cv2.imshow('frame', img1)
    img = img1.copy()
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    print(img1.shape)
    img1 = img1[..., tf.newaxis]
    img1 = resizeImages(img1, 256, 256)
    plt.imshow(img1[:, :, 0])
    print(img1.shape)
    img1 = img1[tf.newaxis, ...]
    print(img1.shape)
    result = model.predict(img1)
    print(result)
    output = result.argmax()
    print(output)
    print(class_labels[result.argmax()])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50) 
  
    # fontScale 
    fontScale = 1

    # Blue color in BGR 
    color = (255, 0, 0) 

    # Line thickness of 2 px 
    thickness = 2
    # Using cv2.putText() method 
    img1 = cv2.putText(img, class_labels[result.argmax()], org, font,  
                       fontScale, color, thickness, cv2.LINE_AA)
    
    if cv2.waitKey(5) == 27:
        break
cv2.destroyAllWindows()
cap.release()
