import os
import cv2
import numpy as np
import pandas as pd
import time
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numba import jit


#@jit(parallel=True)
def get_data():
    df = pd.read_csv('../labeled_photos.csv')
    abs_paths = df['abs_path'].to_list()
    file_names = df['file_name'].to_list()
    given_labels = df['label'].to_list()

    people = []

    for label in given_labels:
        if label not in people:
            people.append(label)

    # change file paths so I can tinker locally
    abs_paths = [path.split('/105_classes_pins_dataset/', 1)[-1] for path in abs_paths]
    prefix = '../data/'
    abs_paths = [prefix + path for path in abs_paths]

    images = []
    labels = []

    for i in range(0, len(abs_paths)):
        image = cv2.imread(abs_paths[i], cv2.IMREAD_GRAYSCALE)
        label = given_labels[i]

        images.append(image)
        labels.append(label)

    images = np.array([image.flatten() for image in images], dtype=object)
    fixed_size = (100, 100)
    images = np.array([np.asarray(Image.fromarray(image).resize(fixed_size)).flatten() for image in images])

    labels = np.array(labels)
    numeric_labels = []
    
    for label in labels:
        numeric_labels.append(people.index(label))

    return images, numeric_labels

if __name__ == '__main__':

    x, y = get_data()
    print('data has been loaded')

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    print('training...')

    start_time = time.time()
    linReg = LinearRegression()
    linReg = linReg.fit(X_train, y_train)
    y_pred = linReg.predict(X_test)

    print("Accuracy: ", accuracy_score(y_test, np.rint(y_pred)))
    print("Time: ", time.time() - start_time)