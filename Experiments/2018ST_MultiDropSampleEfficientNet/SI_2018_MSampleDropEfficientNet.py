from efficientnet import EfficientNetB3, EfficientNetB2, EfficientNetB0
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Input, GlobalAveragePooling2D, Average
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy

from CUtils.SI_DataGenerator2018 import DataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import datetime

# DATASET_PATH = "F:\\Ubuntu\\ISIC2019\\Dataset_19\\"
LAST_YR = "F:\\Ubuntu\\ISIC2018\\ISIC2018_Task3_Training_GroundTruth\\ISIC2018_Task3_Training_GroundTruth.csv"


MODELS_PATH = "F:\\Ubuntu\\ISIC2019\\TrainedModels\\"

# img_width = 718
# img_height = 542
# img_width = 512
# img_height = 360
img_width = 224
img_height = 224

img_shape = (img_width, img_height, 3)


# def top_2_accuracy(y_true, y_pred):
#     return top_k_categorical_accuracy(y_true, y_pred, k=2)


class DataGen:
    def __init__(self, training_cfg):
        self.training_cfg = training_cfg

        # dataframe = pd.read_csv(DATASET_PATH + "ISIC_2019_Training_GroundTruth_Metadata.csv", dtype=str)
        dataframe = pd.read_csv(LAST_YR, dtype=str)

        # anatom_df = pd.get_dummies(dataframe['anatom_site_general'])
        # anatom_df.to_csv(DATASET_PATH + "Anatom", index=False)

        X = dataframe.pop('image')
        # X_train, X_valid, y_train, y_valid = train_test_split(X, dataframe, test_size=0.3)

        train_upto = 6000
        valid_upto = 8000
        # train_upto = 10
        # valid_upto = 15

        X_train = X[0:train_upto]
        X_valid = X[train_upto:valid_upto]
        y_train = dataframe[0:train_upto]
        y_valid = dataframe[train_upto:valid_upto]


        # self.y_anatom = ["anterior torso", "head/neck", "lateral torso", "lower extremity",
        #                 "oral/genital", "palms/soles", "posterior torso", "upper extremity"]
        self.y_anatom = None
        self.y_gen = None
        self.y_age = None

        self.y_cat = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
        # self.y_gen = ["female", "male"]
        # self.y_age = ['age_approx']
        limit_samples = None
        # Generators
        self.training_gen = DataGenerator(Img_IDs=X_train.values,
                                          y_df=y_train,
                                          batch_size=self.training_cfg.batch_size,
                                          x_dim=img_shape,
                                          y_cat_col=self.y_cat,
                                          y_gen_col=self.y_gen,
                                          y_anatom_col=self.y_anatom,
                                          y_age_col=self.y_age,
                                          scale_age=100)

        self.validation_gen = DataGenerator(Img_IDs=X_valid.values,
                                            y_df=y_valid,
                                            batch_size=self.training_cfg.batch_size,
                                            x_dim=img_shape,
                                            y_cat_col=self.y_cat,
                                            y_gen_col=self.y_gen,
                                            y_anatom_col=self.y_anatom,
                                            y_age_col=self.y_age,
                                            scale_age=100,
                                            shuffle=self.training_cfg.shuffle)


class TrainingCfg:
    def __init__(self):
        print(" Training config invoked")
        self.batch_size = 2
        self.nb_epochs = 100
        self.lr = 0.0001
        self.nb_samples = 0
        self.seed = 0
        self.shuffle = True
        self.droupout_list = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.45, 0.5]
        # self.droupout_list = [0.45, 0.5]
        self.optimizer = Adam(lr=self.lr)
        self.metrics = {'cat_pred': ["categorical_accuracy"]}

        self.losses = {'cat_pred': 'categorical_crossentropy'}

        self.loss_weights = {'cat_pred': 1.0}


class MTModel:
    def __init__(self, training_cfg):
        # print(" MT Model invoked")
        self.model = None
        self.training_cfg = training_cfg
        self.build()

    def build(self):
        # print(" MT Model invoked")
        encoder = self.get_encoder(image_shape=img_shape)
        cat_de = self.get_cat_decoder(encoder)

        # gen_de = self.get_gen_decoder(encoder)
        # anatom_de = self.get_anatom_decoder(encoder)
        # age_de = self.get_age_decoder(encoder)
        #
        # gen_en = self.get_gen_encoder(shape=2)
        # anatom_en = self.get_anatom_encoder(shape=8)
        # age_en = self.get_age_encoder(shape=1)

        self.model = Model(encoder.input, cat_de)
        # self.model = Model(encoder.input, [cat_de])
        print(self.model.summary())

    def get_encoder(self, image_shape=img_shape):
        # print(" MT Model invoked")
        encoder = EfficientNetB2(weights='imagenet', include_top=False, input_shape=image_shape)
        return encoder

    # def get_gen_encoder(self, shape):
    #     meta_in = Input(shape=(shape,))
    #     return meta_in
    #
    # def get_anatom_encoder(self, shape):
    #     meta_in = Input(shape=(shape,))
    #     return meta_in
    #
    # def get_age_encoder(self, shape):
    #     meta_in = Input(shape=(shape,))
    #     return meta_in

    def get_multi_sample_droupout(self, pool_out, num_classes, activation, layer_name):
        out = []
        num_neurons = 1408
        for i, drop_rate in enumerate(self.training_cfg.droupout_list):
            drop = Dropout(drop_rate)(pool_out)
            shared_fc1 = Dense(num_neurons, activation='relu')
            fc1 = shared_fc1(drop)
            # shared_fc2 = Dense(num_neurons, activation='relu')
            # fc2 = shared_fc2(fc1)
            # shared_fc3 = Dense(num_classes*100, activation='relu')
            # fc3 = shared_fc3(fc2)
            if '' == activation:
                shared_fc4 = Dense(num_classes, name=layer_name + str(i))
            else:
                shared_fc4 = Dense(num_classes, activation=activation, name=layer_name + str(i))
            fc4 = shared_fc4(fc1)
            out.append(fc4)
        return out

    def get_cat_decoder(self, encoder):
        output_classes = 7
        pool_out = GlobalAveragePooling2D(name='cat_avg_pool')(encoder.output)
        x = self.get_multi_sample_droupout(pool_out, output_classes, 'softmax', 'cat_pred')
        x = Average(name="cat_pred")(x)
        return x

    def get_gen_decoder(self, encoder):
        output_classes = 2
        pool_out = GlobalAveragePooling2D(name='gen_avg_pool')(encoder.output)
        x = self.get_multi_sample_droupout(pool_out, output_classes, 'softmax', 'gen_pred')
        x = Average(name="gen_pred")(x)
        return x

    def get_anatom_decoder(self, encoder):
        output_classes = 8
        pool_out = GlobalAveragePooling2D(name='anatom_avg_pool')(encoder.output)
        x = self.get_multi_sample_droupout(pool_out, output_classes, 'softmax', 'anatom_pred')
        x = Average(name="anatom_pred")(x)
        return x

    def get_age_decoder(self, encoder):
        output_classes = 1
        pool_out = GlobalAveragePooling2D(name='age_avg_pool')(encoder.output)
        x = self.get_multi_sample_droupout(pool_out, output_classes, '', 'age_pred')
        x = Average(name="age_pred")(x)
        return x

    def get_model(self):
        print(" Get Model invoked")
        return self.model


def get_callbacks(app):
    file_path = app.models_path + \
                "weights.E{epoch:02d}-TL{loss:.2f}-TA{categorical_accuracy:.2f}-VA{val_categorical_accuracy:.2f}-VL{val_loss:.2f}.hdf5"
    cp = ModelCheckpoint(filepath=file_path,
                         # monitor='val_loss',
                         verbose=0,
                         # save_best_only=True,
                         save_weights_only=False,
                         mode='auto',
                         period=1)
    log_dir = app.models_path + "TF_logs"
    tb = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)

    return [tb, cp]


class App:
    def __init__(self, mt_model, mt_training_cfg):
        self.model = mt_model.get_model()
        self.train_cfg = mt_training_cfg
        self.datagen = DataGen(mt_training_cfg)
        self.models_path = self.get_folder_path()
        self.model.compile(optimizer=self.train_cfg.optimizer,
                           loss=self.train_cfg.losses,
                           metrics=self.train_cfg.metrics,
                           loss_weights=self.train_cfg.loss_weights)

    def get_folder_path(self):
        folder_path = MODELS_PATH + datetime.datetime.now().strftime("%B_%d_%H_%M_%S")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return folder_path + "\\"

    def train(self):
        print("Starting training")

        self.model.fit_generator(generator=self.datagen.training_gen,
                                 validation_data=self.datagen.validation_gen,
                                 epochs=self.train_cfg.nb_epochs,
                                 workers=4,
                                 max_queue_size=4,
                                 use_multiprocessing=False,
                                 callbacks=get_callbacks(self))

        print("Training Finished")


if __name__ == "__main__":
    training_cfg = TrainingCfg()
    model = MTModel(training_cfg)
    mt_app = App(model, training_cfg)
    mt_app.train()
