import pickle

from skimage.transform import resize
import numpy as np
import cv2
from tensorflow.keras.models import load_model


EMPTY = True
NOT_EMPTY = False
IMAGE_SIZE = (70, 70)

class_names = ['empty', 'not_empty']

MODEL = load_model("classification_model.h5")


def empty_or_not(img):
    # Convert the image to float32 and normalize
    img = img.astype(np.float32) / 255.0
    # Resize the image
    img = cv2.resize(img, IMAGE_SIZE)
    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    predict = MODEL.predict(img)
    predicted_class_index = np.argmax(predict, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]

    return predicted_class_name == 'empty'  # Return boolean instead of string

def get_parking_spots_bboxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components

    slots = []
    for i in range(1, totalLabels):

        # Now extract the coordinate points
        x1 = int(values[i, cv2.CC_STAT_LEFT])
        y1 = int(values[i, cv2.CC_STAT_TOP])
        w = int(values[i, cv2.CC_STAT_WIDTH])
        h = int(values[i, cv2.CC_STAT_HEIGHT])

        slots.append([x1, y1, w, h])

    return slots

