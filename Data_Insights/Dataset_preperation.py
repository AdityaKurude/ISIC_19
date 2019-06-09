import numpy as np
import pandas as pd
import os
from PIL import Image


DATASET_PATH = "F:\\Ubuntu\\ISIC2019\\Dataset_19\\"

def process():
    dataframe = pd.read_csv(DATASET_PATH + "ISIC_2019_Training_GroundTruth_Metadata.csv", dtype=str)
    anatom_df = dataframe['age_approx']
    anatom_df.fillna(0, inplace=True)
    anatom_df.to_csv(DATASET_PATH + "age_approx.csv", index=False)

if __name__ == "__main__":

    process()



