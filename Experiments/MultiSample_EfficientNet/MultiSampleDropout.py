from keras.layers import Input, Dense, Dropout, Average
from keras.models import Model
import numpy as np

data = np.array([[0, 0], [0, 1], [0, 1], [1, 1]])
labels = np.array([[0], [0], [0], [1]])
# labels = np.array([0, 0, 0, 1])

# This returns a tensor
inputs = Input(shape=(2,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(8, activation='relu')(inputs)
x = Dropout(0.5)(x)
x = Dense(8, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['binary_accuracy'])





# pool_out = SomePoolingLayer()(input_tensor)
# shared_fc = Dense(neurons, activation='softmax')
# drop1 = Dropout(0.5)(pool_out)
# drop2 = Droput(0.5)(pool_out)
#
# fc1 = shared_fc(drop1)
# fc2 = shared_fc(drop2)
#
# out = somehow_merge()([fc1, fc2])




# pool_out = Input(shape=(2,))
# drop1 = Dropout(0.2)(pool_out)
# drop2 = Dropout(0.5)(pool_out)
#
# shared_fc = Dense(8, activation='relu')
# shared_fc2 = Dense(8, activation='relu')
# shared_fc3 = Dense(1, activation='sigmoid')
#
# fc10 = shared_fc(drop1)
# fc20 = shared_fc(drop2)
#
# fc11 = shared_fc2(fc10)
# fc21 = shared_fc2(fc20)
#
# fc12 = shared_fc3(fc11)
# fc22 = shared_fc3(fc21)
#
# out = Average()([fc12, fc22])
#
# # This creates a model that includes
# # the Input layer and three Dense layers
# model = Model(inputs=pool_out, outputs=out)
# model.compile(optimizer='adam',
#               loss='mean_squared_error',
#               metrics=['binary_accuracy'])


model.summary()
model.fit(data, labels, epochs=300, batch_size=4)  # starts training

x = np.array([[0, 1], [1, 1], [0, 0], [1, 0]])
result = model.predict(x)

print("result = {} ".format(result))
