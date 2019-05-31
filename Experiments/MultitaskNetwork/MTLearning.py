from keras.applications.xception import Xception
from keras.layers.core import Dense
from keras.models import Model
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Flatten
from keras.layers import Input, GlobalAveragePooling2D
import numpy as np
import pandas as pd
import os


DATASET_PATH = "F:\\Ubuntu\\ISIC2019\\Dataset_19\\"

img_width = 718
img_height = 542
img_shape = (img_width, img_height, 3)



class DataGen:
    def __init__(self, training_cfg):
        self.training_cfg = training_cfg
        self.train_stepsize = 0
        self.validation_stepsize = 0
        self.test_stepsize = 0

        def append_ext(fn):
            return fn + ".jpg"

        traindf = pd.read_csv(DATASET_PATH + "10_Samples_Training_GroundTruth.csv", dtype=str)
        # testdf = pd.read_csv("./sampleSubmission.csv", dtype=str)
        traindf["image"] = traindf["image"].apply(append_ext)
        # testdf["id"] = testdf["id"].apply(append_ext)
        datagen = ImageDataGenerator(rescale=1. / 255., validation_split=0.25)
        location = DATASET_PATH + "//10_Samples//train"
        print("Accessing images at location {}".format(location))
        self.train_generator = datagen.flow_from_dataframe(dataframe=traindf,
                                                      directory=location,
                                                      x_col="image",
                                                      y_col="MEL",
                                                      subset="training",
                                                      batch_size=self.training_cfg.batch_size,
                                                      seed=self.training_cfg.seed,
                                                      shuffle=self.training_cfg.shuffle,
                                                      class_mode="categorical",
                                                      target_size=(img_width, img_height))

        self.valid_generator = datagen.flow_from_dataframe(dataframe=traindf,
                                                      directory=location,
                                                      x_col="image",
                                                      y_col="MEL",
                                                      subset="validation",
                                                      batch_size=self.training_cfg.batch_size,
                                                      seed=self.training_cfg.seed,
                                                      shuffle=self.training_cfg.shuffle,
                                                      class_mode="categorical",
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
        self.metrics = {}
        self.losses = {}
        self.target_img = ()


class MTModel:
    def __init__(self):
        print(" MT Model invoked")
        self.model = None
        self.build()

    def build(self):
        print(" MT Model invoked")
        encoder = self.get_encoder(image_shape=img_shape)
        decoder = self.get_decoder(encoder, output_classes=2)
        self.model = Model(encoder.input, decoder)
        # print(self.model.summary())

    def get_encoder(self, image_shape=img_shape):
        print(" MT Model invoked")
        encoder = Xception(weights='imagenet', include_top=False, input_shape=image_shape)
        return encoder

    def get_decoder(self, encoder, output_classes = 6):
        x = GlobalAveragePooling2D(name='avg_pool')(encoder.output)
        x = Dense(output_classes, activation='softmax', name='predictions')(x)
        return x

    def get_model(self):
        print(" MT Model invoked")
        return self.model


class App:
    def __init__(self, mt_model, mt_training_cfg):
        self.model = mt_model.get_model()
        self.model.compile(optimizer='adam',
                  loss='categorical_crossentropy',  # categorical_crossentropy if multi-class classifier
                  metrics=['accuracy'])
        self.train_cfg = mt_training_cfg
        self.datagen = DataGen(mt_training_cfg)

    def train(self):
        print("Starting training")

        self.model.fit_generator(generator=self.datagen.train_generator,
                                 steps_per_epoch=self.datagen.train_stepsize,
                                 validation_data=self.datagen.valid_generator,
                                 validation_steps=5,
                                 epochs=5)

        print("Training Finished")



if __name__ == "__main__":
    model = MTModel()
    training_cfg = TrainingCfg()

    mt_app = App(model, training_cfg)
    mt_app.train()
