import numpy as np
import pandas as pd
import keras
import cv2


DATASET_PATH = "F:\\Ubuntu\\ISIC2019\\Dataset_19\\"
IMAGE_PATH = "F:\\Ubuntu\\ISIC2019\\Dataset_19\\10_Samples\\train\\"

class DataGenerator(keras.utils.Sequence):
    # 'Generates data for Keras'
    def __init__(self, Img_IDs, y_df, batch_size=1, x_dim=(718, 542, 3), y_col=[], shuffle=True):
        'Initialization'
        self.x_dim = x_dim
        self.y_col = y_col
        self.batch_size = batch_size
        self.Img_IDs = Img_IDs
        self.shuffle = shuffle
        self.dataframe = y_df
        self.col_index = []
        self.on_epoch_end()
        self.get_column_index()

    def get_column_index(self):
        col_l = list(self.dataframe.columns.values)
        print(" All columns present in the dataframe are \n {} ".format(col_l))
        found_idx = 0
        for idx, col in enumerate(col_l):
            if len(self.y_col) == found_idx:
                return
            if self.y_col[found_idx] == col:
                found_idx = found_idx + 1
                self.col_index.append(idx)
                print(" Found column {} at index {} ".format(col, idx))

    def __len__(self):
        # 'Denotes the number of batches per epoch'
        return int(np.floor(len(self.Img_IDs) / self.batch_size))

    def __getitem__(self, index):
        # 'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        Img_IDs_temp = [self.Img_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(Img_IDs_temp, index)

        return X, y

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.Img_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, Img_IDs_temp, index):
        # 'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.x_dim))
        y = np.empty((self.batch_size, len(self.y_col)), dtype=int)
        print("data generator list is ", Img_IDs_temp)

        # Generate data
        for i, ID in enumerate(Img_IDs_temp):
            # Store sample Images
            img = cv2.imread(IMAGE_PATH + ID + '.jpg')
            # TODO: check this resizing and cropping options
            res = cv2.resize(img, dsize=(self.x_dim[1], self.x_dim[0]), interpolation=cv2.INTER_NEAREST)
            X[i, ] = res

            # Store class
            y[i, ] = self.dataframe.iloc[(index + i), self.col_index].values

            print(" Read img {} and col values {} ".format((IMAGE_PATH + ID), y[i, :]))

        # print("generator returning values in shape train-val", train_subset.values.shape,y.shape)
        return X, y