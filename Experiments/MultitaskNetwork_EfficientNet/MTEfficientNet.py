from efficientnet import EfficientNetB3
from keras.layers.core import Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Input, GlobalAveragePooling2D
from CUtils.MT_DataGenerator import DataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import datetime


DATASET_PATH = "F:\\Ubuntu\\ISIC2019\\Dataset_19\\"
MODELS_PATH = "F:\\Ubuntu\\ISIC2019\\TrainedModels\\"

# img_width = 718
# img_height = 542
img_width = 512
img_height = 320
img_shape = (img_width, img_height, 3)


class DataGen:
    def __init__(self, training_cfg):
        self.training_cfg = training_cfg

        dataframe = pd.read_csv(DATASET_PATH + "ISIC_2019_Training_GroundTruth_Metadata.csv", dtype=str)
        # anatom_df = pd.get_dummies(dataframe['anatom_site_general'])
        # anatom_df.to_csv(DATASET_PATH + "Anatom", index=False)

        X = dataframe.pop('image')
        X_train, X_valid, y_train, y_valid = train_test_split(X, dataframe, test_size=0.8)

        y_anatom = ["anterior torso", "head/neck", "lateral torso", "lower extremity",
                        "oral/genital", "palms/soles", "posterior torso", "upper extremity"]

        y_cat = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
        y_gen = ["female", "male"]

        # Generators
        self.training_gen = DataGenerator(Img_IDs=X_train.values,
                                           y_df=y_train,
                                           batch_size=1,
                                           x_dim=(512,320,3),
                                           y_cat_col=y_cat,
                                          y_gen_col=y_gen,
                                          y_anatom_col=y_anatom,
                                          y_age_col=['age_approx'],
                                          scale_age=10)

        self.validation_gen = DataGenerator(Img_IDs=X_valid.values,
                                           y_df=y_valid,
                                           batch_size=1,
                                           x_dim=(512,320,3),
                                            y_cat_col=y_cat,
                                            y_gen_col=y_gen,
                                            y_anatom_col=y_anatom,
                                            y_age_col=['age_approx'],
                                            scale_age=10)


class TrainingCfg:
    def __init__(self):
        print(" Training config invoked")
        self.batch_size = 2
        self.nb_epochs = 10
        self.lr = 0.001
        self.nb_samples = 0
        self.seed = 0
        self.shuffle = False
        self.optimizer = 'adam'
        self.metrics = {'cat_pred': 'accuracy',
                        'gen_pred': 'accuracy',
                        'anatom_pred': 'accuracy',
                        'age_pred': ['mae', 'accuracy']}

        self.losses = {'cat_pred': 'categorical_crossentropy',
                       'gen_pred': 'categorical_crossentropy',
                       'anatom_pred': 'categorical_crossentropy',
                       'age_pred': 'mean_squared_error'}

        self.loss_weights = {'cat_pred': 1.0,
                             'gen_pred': 1.0,
                             'anatom_pred': 1.0,
                             'age_pred': 0.1}

class MTModel:
    def __init__(self):
        # print(" MT Model invoked")
        self.model = None
        self.build()

    def build(self):
        # print(" MT Model invoked")
        encoder = self.get_encoder(image_shape=img_shape)
        cat_de = self.get_cat_decoder(encoder)
        gen_de = self.get_gen_decoder(encoder)
        anatom_de = self.get_anatom_decoder(encoder)
        age_de = self.get_age_decoder(encoder)

        self.model = Model(encoder.input, [cat_de, gen_de, anatom_de, age_de])
        # self.model = Model(encoder.input, [cat_de])
        # print(self.model.summary())

    def get_encoder(self, image_shape=img_shape):
        # print(" MT Model invoked")
        encoder = EfficientNetB3(weights='imagenet', include_top=False, input_shape=image_shape)
        return encoder

    def get_cat_decoder(self, encoder):
        output_classes = 8
        x = GlobalAveragePooling2D(name='cat_avg_pool')(encoder.output)
        x = Dense(output_classes, activation='softmax', name='cat_pred')(x)
        return x

    def get_gen_decoder(self, encoder):
        output_classes = 2
        x = GlobalAveragePooling2D(name='gen_avg_pool')(encoder.output)
        x = Dense(output_classes, activation='softmax', name='gen_pred')(x)
        return x

    def get_anatom_decoder(self, encoder):
        output_classes = 8
        x = GlobalAveragePooling2D(name='anatom_avg_pool')(encoder.output)
        x = Dense(output_classes, activation='softmax', name='anatom_pred')(x)
        return x

    def get_age_decoder(self, encoder):
        output_classes = 1
        x = GlobalAveragePooling2D(name='age_avg_pool')(encoder.output)
        x = Dense(output_classes*10, name='age_pre_pred')(x)
        x = Dense(output_classes, name='age_pred')(x)
        return x

    def get_model(self):
        print(" MT Model invoked")
        return self.model


def get_folder_path():
    folder_path = MODELS_PATH + datetime.datetime.now().strftime("%B_%d")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path


def get_callbacks():
    file_path = get_folder_path() + "\\" + datetime.datetime.now().strftime('%H_%M_%S')[:-4] \
                + "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    cp = ModelCheckpoint(filepath=file_path,
                         monitor='val_loss',
                         verbose=0,
                         save_best_only=True,
                         save_weights_only=False,
                         mode='auto',
                         period=1)
    log_dir = get_folder_path() + "\\" + datetime.datetime.now().strftime('%H_%M_%S')[:-4]
    tb = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)

    return [tb, cp]


class App:
    def __init__(self, mt_model, mt_training_cfg):
        self.model = mt_model.get_model()
        self.train_cfg = mt_training_cfg
        self.datagen = DataGen(mt_training_cfg)
        self.model.compile(optimizer=self.train_cfg.optimizer,
                           loss=self.train_cfg.losses,
                           metrics=self.train_cfg.metrics,
                           loss_weights=self.train_cfg.loss_weights)

    def train(self):
        print("Starting training")

        self.model.fit_generator(generator=self.datagen.training_gen,
                                 validation_data=self.datagen.validation_gen,
                                 epochs=self.train_cfg.nb_epochs,
                                 workers=1,
                                 max_queue_size=2,
                                 use_multiprocessing=False,
                                 callbacks=get_callbacks())

        print("Training Finished")


if __name__ == "__main__":
    model = MTModel()
    training_cfg = TrainingCfg()

    mt_app = App(model, training_cfg)
    mt_app.train()