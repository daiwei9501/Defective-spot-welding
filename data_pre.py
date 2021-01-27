import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

folder = 'spot/'
img_data = []
img_label = []

dict = {'good': 0, "edge welds": 1, 'double': 2,
        'pseudo welds':3, 'burr':4, 'distortion':5, 'burn through':6}

m = 64
n = 64

for class_name in os.listdir(folder):
    for image_name in os.listdir(os.path.join(folder, class_name)):
        imgpath = os.path.join(folder, class_name, image_name)
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (m, n))
        img_data.append(img)
        img_label.append(dict[class_name])

image_data = np.array(img_data)
image_label = np.array(img_label)
image_label = image_label[:, np.newaxis]

x_train, x_val, y_train, y_val = train_test_split(image_data, image_label, test_size=0.1, stratify=image_label, random_state=1)
print(x_train.shape)
print(y_val.shape)

# Save
np.save("x_train.npy", x_train)
np.save("y_train.npy", y_train)
np.save("x_val.npy", x_val)
np.save("y_val.npy", y_val)