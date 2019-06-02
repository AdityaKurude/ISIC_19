from keras.applications.xception import Xception
from keras.layers.core import Dense
from keras.models import Model
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, GlobalAveragePooling2D
import numpy as np
import pandas as pd
import os
import datetime


DATASET_PATH = "F:\\Ubuntu\\ISIC2019\\Dataset_19\\"
MODELS_PATH = "F:\\Ubuntu\\ISIC2019\\TrainedModels\\"

img_width = 718
img_height = 542
img_shape = (img_width, img_height, 3)


def generator_wrapper(generator):
    for batch_x,batch_y in generator:
        yield (batch_x,[batch_y[:, i] for i in range(2)])


class DataGen:
    def __init__(self, training_cfg):
        self.training_cfg = training_cfg
        self.train_stepsize = 0
        self.validation_stepsize = 0
        self.test_stepsize = 0

        def append_ext(fn):
            return fn + ".jpg"

        traindf = pd.read_csv(DATASET_PATH + "10_Samples_Training_GroundTruth_Mod.csv", dtype=str)
        # testdf = pd.read_csv("./sampleSubmission.csv", dtype=str)
        traindf["image"] = traindf["image"].apply(append_ext)
        # testdf["id"] = testdf["id"].apply(append_ext)
        datagen = ImageDataGenerator(rescale=1. / 255., validation_split=0.3)
        location = DATASET_PATH + "\\10_Samples\\train"
        print("Accessing images at location {}".format(location))
        column_y = ["Merged", "gender"]

        if "object" in list(traindf[column_y].dtypes):

            for i in range(len(list(traindf[column_y].dtypes))):
                print(" Numeric hence converting coloumn {}".format(i))
                traindf[column_y[i]] = pd.to_numeric(traindf[column_y[i]])

        self.train_generator = datagen.flow_from_dataframe(dataframe=traindf,
                                                           directory=location,
                                                           x_col="image",
                                                           y_col=column_y,
                                                           subset="training",
                                                           batch_size=self.training_cfg.batch_size,
                                                           seed=self.training_cfg.seed,
                                                           shuffle=self.training_cfg.shuffle,
                                                           class_mode="other",
                                                           target_size=(img_width, img_height))

        self.valid_generator = datagen.flow_from_dataframe(dataframe=traindf,
                                                           directory=location,
                                                           x_col="image",
                                                           y_col=column_y,
                                                           subset="validation",
                                                           batch_size=self.training_cfg.batch_size,
                                                           seed=self.training_cfg.seed,
                                                           shuffle=self.training_cfg.shuffle,
                                                           class_mode="other",
                                                           target_size=(img_width, img_height))

        # test_datagen = ImageDataGenerator(rescale=1. / 255.)
        # self.test_generator = test_datagen.flow_from_dataframe(
        #     dataframe=testdf,
        #     directory="./test/",
        #     x_col="id",
        #     y_col=None,
        #     batch_size=32,
        #     seed=42,
        #     shuffle=False,
        #     class_mode=None,
        #     target_size=(32, 32))

        self.train_stepsize = self.train_generator.n // self.train_generator.batch_size
        self.validation_stepsize = self.valid_generator.n // self.valid_generator.batch_size
        # self.test_stepsize = self.test_generator.n // self.test_generator.batch_size

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
                        'gen_pred': 'accuracy'}

        self.losses = {'cat_pred': 'binary_crossentropy',
                       'gen_pred': 'binary_crossentropy'}

        self.loss_weights = {'cat_pred': 1.0,
                             'gen_pred': 1.0}

        self.target_img = ()


class MTModel:
    def __init__(self):
        print(" MT Model invoked")
        self.model = None
        self.build()

    def build(self):
        print(" MT Model invoked")
        encoder = self.get_encoder(image_shape=img_shape)
        cat_de = self.get_cat_decoder(encoder)
        gen_de = self.get_gen_decoder(encoder)
        self.model = Model(encoder.input, [cat_de, gen_de])
        # print(self.model.summary())

    def get_encoder(self, image_shape=img_shape):
        print(" MT Model invoked")
        encoder = Xception(weights='imagenet', include_top=False, input_shape=image_shape)
        return encoder

    def get_cat_decoder(self, encoder):
        output_classes = 1
        x = GlobalAveragePooling2D(name='cat_avg_pool')(encoder.output)
        x = Dense(output_classes, activation='softmax', name='cat_pred')(x)
        return x

    def get_gen_decoder(self, encoder):
        output_classes = 1
        x = GlobalAveragePooling2D(name='gen_avg_pool')(encoder.output)
        x = Dense(output_classes, activation='softmax', name='gen_pred')(x)
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
                         save_best_only=False,
                         save_weights_only=False,
                         mode='auto',
                         period=1)

    return [cp]


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

        self.model.fit_generator(generator=generator_wrapper(self.datagen.train_generator),
                                 steps_per_epoch=self.datagen.train_stepsize,
                                 validation_data=generator_wrapper(self.datagen.valid_generator),
                                 validation_steps=5,
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
