from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Input, Average
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)







inputs = Input(shape=input_shape)
x = Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape)(inputs)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)






# inputs = Input(shape=input_shape)
# x = Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape)(inputs)
# x = Conv2D(64, (3, 3), activation='relu')(x)
# pool_out = MaxPooling2D(pool_size=(2, 2))(x)
# drop1 = Dropout(0.1)(pool_out)
# drop2 = Dropout(0.2)(pool_out)
# drop3 = Dropout(0.35)(pool_out)
# drop4 = Dropout(0.40)(pool_out)
# drop5 = Dropout(0.45)(pool_out)
# drop6 = Dropout(0.25)(pool_out)
# drop7 = Dropout(0.15)(pool_out)
# drop8 = Dropout(0.5)(pool_out)
#
# shared_flat = Flatten()
# shared_fc2 = Dense(128, activation='relu')
# shared_fc3 = Dense(num_classes, activation='softmax')
#
# fc10 = shared_flat(drop1)
# fc20 = shared_flat(drop2)
# fc30 = shared_flat(drop3)
# fc40 = shared_flat(drop4)
# fc50 = shared_flat(drop5)
# fc60 = shared_flat(drop6)
# fc70 = shared_flat(drop7)
# fc80 = shared_flat(drop8)
#
# fc11 = shared_fc2(fc10)
# fc21 = shared_fc2(fc20)
# fc31 = shared_fc2(fc10)
# fc41 = shared_fc2(fc20)
# fc51 = shared_fc2(fc10)
# fc61 = shared_fc2(fc20)
# fc71 = shared_fc2(fc10)
# fc81 = shared_fc2(fc20)
#
# fc12 = shared_fc3(fc11)
# fc22 = shared_fc3(fc21)
# fc32 = shared_fc3(fc11)
# fc42 = shared_fc3(fc21)
# fc52 = shared_fc3(fc11)
# fc62 = shared_fc3(fc21)
# fc72 = shared_fc3(fc11)
# fc82 = shared_fc3(fc21)
#
# out = Average()([fc12, fc22, fc32, fc42, fc52, fc62, fc72, fc82])
#
# # This creates a model that includes
# # the Input layer and three Dense layers
# model = Model(inputs=inputs, outputs=out)





model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


import tensorflow as tf
import keras.backend as K


def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.

print(" FLOPS of the model \n ")
print(get_flops(model))
exit()


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



