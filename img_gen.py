import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib.pylab as plt
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

images = np.load('x_train.npy')
labels = np.load('y_train.npy')

######################## Preprocessing ##########################
# Set channel
channel = images.shape[-1]

# to 64 x 64 x channel
real = np.ndarray(shape=(images.shape[0], 64, 64, channel))
for i in range(images.shape[0]):
    real[i] = cv2.resize(images[i], (64, 64)).reshape((64, 64, channel))

# Train test split, for autoencoder (actually, this step is redundant if we already have test set)
x_train, x_test, y_train, y_test = train_test_split(real, labels, test_size=0.3, shuffle=True, random_state=42)

# It is suggested to use [-1, 1] input for GAN training
x_train = (x_train.astype('float32') - 127.5) / 127.5
x_test = (x_test.astype('float32') - 127.5) / 127.5

# Get image size
img_size = x_train[0].shape
# Get number of classes
n_classes = len(np.unique(y_train))

# %% ---------------------------------- Hyperparameters ----------------------------------------------------------------

optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.9)
latent_dim=128
# trainRatio === times(Train D) / times(Train G)
trainRatio = 5
d_model = load_model('d_model_8.h5')
g_model = load_model('g_model_8.h5')

def vis_square(generator, padsize=1, padval=0):
    # generate some pictures to display
    np.random.seed(1)
    noise = np.random.normal(size= (7 * n_classes, latent_dim))
    sampled_labels = np.array([[i] * 7 for i in range(n_classes)]).reshape(-1, 1)
    # generated_images = generator.predict([noise, sampled_labels]).transpose(0, 2, 3, 1)
    data = generator.predict([noise, sampled_labels])
    data = data * 127.5 + 127.5
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    #Image.fromarray(data).save('plot_generated.png')
    return data

def gene_img(generator, label, num_img):
    for i in range(num_img):
        noise = np.random.normal(size=(1, latent_dim))
        sampled_labels = np.array([label])
        data = generator.predict([noise, sampled_labels])
        data = data * 127.5 + 127.5
        data = data.reshape(data.shape[1:])
        Image.fromarray(np.uint8(data)).save('images/' + str(label) + '/' + str(i) + '.png')

# if __name__=="__main__":
#     for i in range(7):
#         gene_img(g_model, i, 2000)

data = vis_square(g_model)
Image.fromarray(np.uint8(data)).save('plot_generated.png')
