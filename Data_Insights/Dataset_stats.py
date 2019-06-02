import numpy as np
import pandas as pd
import os
from PIL import Image


DATASET_PATH = "F:\\Ubuntu\\ISIC2019\\Dataset_19\\"

dataset_gt = pd.read_csv( DATASET_PATH + "ISIC_2019_Training_GroundTruth.csv", delimiter = ',')

merged_list = np.zeros(shape=(25331,1), dtype=int)

def get_stats():
    col_num = 1
    for col_head in dataset_gt.head(1):
        if not col_head == 'image':
            # print(col_head)
            col = dataset_gt[col_head].to_numpy()
            n_total_col = len(col)
            # Since results are binary, simple sum gives all positive detections
            n_detections_col = np.sum(col)
            print("{0}. {1} total = {2} and Positive detections {3}".format(col_num, col_head, n_total_col, n_detections_col))

            # Section create simple multi-class dataset
            merged_list[col == 1] = 1
            col_num = col_num + 1

    # merged_list.astype(int)
    np.savetxt("bin_foo.csv", merged_list, delimiter=",", fmt='%i')


def check_multilabel():
    is_multilabel = False
    for row in dataset_gt.itertuples():
        row_t = np.sum(row[2:])
        row_arr = np.asarray(row_t)
        sum = np.sum(row_arr)
        # print(sum)
        # print(row[2:])
        if sum > 1.0:
            is_multilabel = True

    print(" The problem is multilabel ? {}".format(is_multilabel))

def get_image_sizes():
    datset = DATASET_PATH + "ISIC_2019_Training_Input\\"
    images = os.listdir(datset)
    # print(images[:10])

    size_variations = list()
    for image_n in images:
        img = Image.open(datset+image_n)
        size = img.size
        # print(size)
        if size not in size_variations:
            size_variations.append(size)

    print(size_variations)
    print("total number of variations {}".format(len(size_variations)))

if __name__ == "__main__":

    get_stats()
    # check_multilabel()
    # get_image_sizes()



