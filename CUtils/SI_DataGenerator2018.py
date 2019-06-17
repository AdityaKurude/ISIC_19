import numpy as np
import pandas as pd
import keras
import cv2


# DATASET_PATH = "F:\\Ubuntu\\ISIC2019\\Dataset_19\\"
# IMAGE_PATH = "F:\\Ubuntu\\ISIC2019\\Dataset_19\\ISIC_2019_Training_Input\\"

DATASET_PATH = "F:\\Ubuntu\\ISIC2019\\Dataset_19\\"
IMAGE_PATH = "F:\\Ubuntu\\ISIC2018\\ISIC2018_Task3_Training_Input\\"


class DataGenerator(keras.utils.Sequence):
    # 'Generates data for Keras'
    def __init__(self, Img_IDs, y_df, batch_size=1, x_dim=(718, 542, 3),
                 y_cat_col=[],
                 y_fine_cat_col=[],
                 y_coarse_cat_col=[],
                 y_age_col=[],
                 scale_age=0,
                 shuffle=False):
        'Initialization'
        self.x_dim = x_dim
        self.scale_age = scale_age

        self.batch_size = batch_size
        self.Img_IDs = Img_IDs
        self.shuffle = shuffle
        self.dataframe = y_df

        self.y_cat_col = y_cat_col
        self.y_cat_index = []

        self.y_fine_cat_col = y_fine_cat_col
        self.y_fine_cat_index = []

        self.y_coarse_cat_col = y_coarse_cat_col
        self.y_coarse_cat_index = []

        self.y_age_col = y_age_col
        self.y_age_index = []

        self.on_epoch_end()
        self.get_column_index()

    def get_column_index(self):
        col_l = list(self.dataframe.columns.values)
        print(" All columns present in the dataframe are \n {} ".format(col_l))

        found_idx = 0
        for idx, col in enumerate(col_l):
            if len(self.y_coarse_cat_col) == found_idx:
                break
            if self.y_coarse_cat_col[found_idx] == col:
                found_idx = found_idx + 1
                self.y_coarse_cat_index.append(idx)
                print(" Found coarse cat column {} at index {} ".format(col, idx))
        # End

        # Iterate for Fine cat categories
        found_idx = 0
        for idx, col in enumerate(col_l):
            if len(self.y_fine_cat_col) == found_idx:
                break
            if self.y_fine_cat_col[found_idx] == col:
                found_idx = found_idx + 1
                self.y_fine_cat_index.append(idx)
                print(" Found Fine Cat column {} at index {} ".format(col, idx))
        # End

        found_idx = 0
        for idx, col in enumerate(col_l):
            if len(self.y_cat_col) == found_idx:
                return
            if self.y_cat_col[found_idx] == col:
                found_idx = found_idx + 1
                self.y_cat_index.append(idx)
                print(" Found cat column {} at index {} ".format(col, idx))
        # End

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
        X, y, y_fine_cat, y_coarse_cat = self.__data_generation(Img_IDs_temp, index*self.batch_size)

        return X,  {'cat_pred': y, "fine_cat_pred": y_fine_cat, "coarse_cat_pred": y_coarse_cat}

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.Img_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, Img_IDs_temp, index):
        # 'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.x_dim))
        y = np.empty((self.batch_size, len(self.y_cat_col)), dtype=int)
        y_fine_cat = np.empty((self.batch_size, len(self.y_fine_cat_col) + 1), dtype=int)
        y_coarse_cat = np.empty((self.batch_size, len(self.y_coarse_cat_col) + 1), dtype=int)
        # y_age = np.zeros((self.batch_size, len(self.y_age_col)), dtype=float)
        # print("data generator list is {} \n".format(Img_IDs_temp))

        # Generate data
        for i, ID in enumerate(Img_IDs_temp):
            # print("Image picked is {} and index {} \n".format(ID, index+i))
            # Store sample Images
            img = cv2.imread(IMAGE_PATH + ID + '.jpg')
            # TODO: check this resizing and cropping options
            res = cv2.resize(img, dsize=(self.x_dim[1], self.x_dim[0]), interpolation=cv2.INTER_NEAREST)
            X[i, ] = res

            tmp = int(index + i)
            print("value = {}".format(tmp))
            # Store class
            np_arr = self.dataframe.iloc[tmp, self.y_cat_index].values
            y[i, ] = np_arr.astype(np.float)

            np_arr = self.dataframe.iloc[(index + i), self.y_fine_cat_index].values
            is_fine_cat = np.sum(np_arr.astype(np.float))

            np_arr = self.dataframe.iloc[(index + i), self.y_fine_cat_index].values
            y_fine_cat[i, ] = \
                np.append( np_arr.astype(np.float), 1 - is_fine_cat)

            np_arr = self.dataframe.iloc[(index + i), self.y_coarse_cat_index].values
            y_coarse_cat[i, ] = \
                np.append(np_arr.astype(np.float), is_fine_cat)
            # print(" Read img {} y {} fine_cat {} coarse_cat {} \n ".format(ID, y[i, :], y_fine_cat[i, :], y_coarse_cat[i, :]))

        return X, y, y_fine_cat, y_coarse_cat