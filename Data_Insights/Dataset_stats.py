import numpy as np
import pandas as pd

dataset_gt = pd.read_csv("F:\\Ubuntu\\ISIC2019\\Dataset_19\\ISIC_2019_Training_GroundTruth.csv", delimiter = ',')

col_num = 1
for col_head in dataset_gt.head(1):
    if not col_head == 'image':
        # print(col_head)
        col = dataset_gt[col_head].to_numpy()
        n_total_col = len(col)
        # Since results are binary, simple sum gives all positive detections
        n_detections_col = np.sum(col)
        print("{0}. {1} total = {2} and Positive detections {3}".format(col_num, col_head, n_total_col, n_detections_col))
        col_num = col_num + 1


