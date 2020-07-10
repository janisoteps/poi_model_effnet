import numpy as np
from PIL import Image, ImageChops


class DataGenerator(object):
    'Generates data for Keras'

    def __init__(self, dim_x=150, dim_y=150, batch_size=16, shuffle=True):
        'Initialization'
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def generate(self, labels, list_IDs):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(list_IDs)

            # Generate batches
            imax = int(len(indexes) / self.batch_size)
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = [list_IDs[k] for k in indexes[i * self.batch_size:(i + 1) * self.batch_size]]

                # Generate data
                X, y = self.__data_generation(labels, list_IDs_temp)

                # X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[3], X_train.shape[4])

                X = X.reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[3])
                # print('x shape: ', str(X.shape))
                # print('y shape: ', str(y.shape))
                yield X, y

    def __get_exploration_order(self, list_IDs):
        'Generates order of exploration'
        # Find exploration order
        print('Shuffling data')
        indexes = np.arange(len(list_IDs))
        if self.shuffle == True:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, labels, list_IDs_temp):
        'Generates data of batch_size samples'  # X : (n_samples, v_size, v_size, v_size, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim_x, self.dim_y, 3))
        y = np.empty((self.batch_size, 2), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store volume
            # print('Create X for: ', str(ID))
            # X[i, :, :, :] = load_image('/Users/jdo/dev/scrapers/scraper11/scraper11/spiders/images/full/' + ID + '.jpg')
            X[i, :, :, :] = load_image(ID)

            # Store class
            print('Storing class for: ', str(ID))
            print(labels[ID])
            y[i] = labels[ID]

        return X, y


def load_image(img_path):
    with open(img_path, 'r+b') as f:
        with Image.open(f) as picture:
            # bg = Image.new(picture.mode, picture.size, picture.getpixel((5, 5)))
            # diff = ImageChops.difference(picture, bg)
            # diff = ImageChops.add(diff, diff, 2.0, -100)
            # bbox = diff.getbbox()
            # pic_cropped = picture.crop(bbox)
            # pic = pic_cropped.resize((299, 299), Image.ANTIALIAS)
            pic = picture.resize((150, 150), Image.ANTIALIAS)
            x = np.array(pic)
            x = (x - 255) / 255
            x = np.expand_dims(x, axis=0)
            # print('Returning img array for: ', img_path)
            return x
