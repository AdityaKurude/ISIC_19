import numpy as np
import pandas as pd
import keras
import cv2


DATASET_PATH = "F:\\Ubuntu\\ISIC2019\\Dataset_19\\"
IMAGE_PATH = "F:\\Ubuntu\\ISIC2019\\Dataset_19\\ISIC_2019_Training_Input\\"

class DataGenerator(keras.utils.Sequence):
    # 'Generates data for Keras'
    def __init__(self, Img_IDs, y_df, batch_size=1, x_dim=(718, 542, 3),
                 y_cat_col=[],
                 y_gen_col=[],
                 y_anatom_col=[],
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

        self.y_gen_col = y_gen_col
        self.y_gen_index = []

        self.y_anatom_col = y_anatom_col
        self.y_anatom_index = []

        self.y_age_col = y_age_col
        self.y_age_index = []

        self.on_epoch_end()
        self.get_column_index()

    def get_column_index(self):
        col_l = list(self.dataframe.columns.values)
        print(" All columns present in the dataframe are \n {} ".format(col_l))

        # Iterate for Age categories
        for idx, col in enumerate(col_l):
            if self.y_age_col[0] == col:
                self.y_age_index.append(idx)
                print(" Found Age column {} at index {} ".format(col, idx))
                break
        # End

        # Iterate for Anatomy categories
        found_idx = 0
        for idx, col in enumerate(col_l):
            if len(self.y_anatom_col) == found_idx:
                break
            if self.y_anatom_col[found_idx] == col:
                found_idx = found_idx + 1
                self.y_anatom_index.append(idx)
                print(" Found Anatom column {} at index {} ".format(col, idx))
        # End

        # Iterate for Gender
        found_idx = 0
        for idx, col in enumerate(col_l):
            if len(self.y_gen_col) == found_idx:
                break
            if self.y_gen_col[found_idx] == col:
                found_idx = found_idx + 1
                self.y_gen_index.append(idx)
                print(" Found gen column {} at index {} ".format(col, idx))
        # End
        # Iterate for Cancer categories
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
        X, y, y_gen, y_anatom, y_age = self.__data_generation(Img_IDs_temp, index)

        return X,  {'cat_pred': y}

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
        y_gen = np.empty((self.batch_size, len(self.y_gen_col)), dtype=int)
        y_anatom = np.empty((self.batch_size, len(self.y_anatom_col)), dtype=int)
        y_age = np.zeros((self.batch_size, len(self.y_age_col)), dtype=float)
        # print("data generator list is {} \n".format(Img_IDs_temp))

        # Generate data
        for i, ID in enumerate(Img_IDs_temp):
            # print("Image picked is {} \n".format(ID))
            # Store sample Images
            img = cv2.imread(IMAGE_PATH + ID + '.jpg')
            # TODO: check this resizing and cropping options
            res = cv2.resize(img, dsize=(self.x_dim[1], self.x_dim[0]), interpolation=cv2.INTER_NEAREST)
            X[i, ] = res
            # Store class
            y[i, ] = self.dataframe.iloc[(index + i), self.y_cat_index].values
            y_gen[i, ] = self.dataframe.iloc[(index + i), self.y_gen_index].values
            y_anatom[i, ] = self.dataframe.iloc[(index + i), self.y_anatom_index].values
            y_age[i, ] = self.dataframe.iloc[(index + i), self.y_age_index].values
            # print(" Read img {} gender {} and col values {} \n ".format(ID, y_gen[i, :], y[i, :]))
            if not self.scale_age == 0:
                y_age[i, ] = np.divide(y_age[i, ], self.scale_age)

            print(" Read img {} Age {} and anatom values {} \n ".format(ID, y_age[i, :], y_anatom[i, :]))

        return X, y, y_gen, y_anatom, y_age