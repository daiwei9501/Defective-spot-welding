import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from keras.models import load_model
from keras.models import Model

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

labels = ['edge weld', 'double', 'pseudo welds', 'burr', 'distortion', 'burn through', 'good']
colors = ['#008000','#0000FF','#FF69B4','#FFA500','#FF0000','#FFFF00','#EE82EE']

x_test, y_test = np.load('x_val.npy'), np.load('y_val.npy')

n_classes = len(np.unique(y_test))
inputShape = x_test[0].shape

# preprocess = imagenet_utils.preprocess_input
# x_train = preprocess(x_train)
# x_test = preprocess(x_test)


# Get the feature output from the pre-trained model ResNet50
#pretrained_model = load_model('./transfer_resnet50model.h5')
pretrained_model = load_model('models/transefer_resnet50.h5')

for layer in pretrained_model.layers:
    layer.trainable = False
x = pretrained_model.layers[-1].output
#x = GlobalAveragePooling2D()(x)
feature_model = Model(pretrained_model.input, x)


# %% -------------------------------------- TSNE Visualization ---------------------------------------------------------
def tsne_plot(feature, y_label):
    "Creates and TSNE model and plots it"
    plt.figure(figsize=(8, 8))
    color = plt.get_cmap('tab10')

    tsne_model = TSNE(n_components=2, init='random', random_state=0)
    new_values = tsne_model.fit_transform(feature)

    # plt.scatter(new_values[:, 0], new_values[:, 1], c=colors[y_label], label=labels[y_label])
    # plt.legend()
    # plt.show()

    x1 = new_values[:,0]
    x2 = new_values[:,1]
    for c in range(n_classes):
        plt.scatter(x1[y_label == c], x2[y_label == c], c=np.array([color(c)]), label=labels[c])
    plt.legend()
    plt.show()

features = feature_model.predict(x_test)
tsne_plot(features, y_test.reshape(-1))