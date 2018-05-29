from __future__ import print_function, division
import pattern_recog_func as prf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2

import numpy as np

if __name__ == "__main__":
    names_dict = {2: "Luke", 0: "Gilbert", 1: "Janet"}
    
    print("Loading images...")
    
    images, y = prf.load_images("svm_training_photos")
    
    images = np.array(images)
    y = np.array(y)
    
    print("Cropping and interpoolating images...")
    
    for i in range(len(images)):
        images[i] = prf.crop_and_interpool_image(images[i])

    X = np.vstack(images)

    print("Starting leave one out test...\n")

    correct = 0
    for i in range(len(X)):
        guess = prf.leave_one_out_test(X, y, i)
        answer = y[i]
        if(guess == answer):
            correct += 1

    print("Correct rate: {} \n".format((correct/(len(X))*100)))

    prf.detect_three_faces(X, y, names_dict)
