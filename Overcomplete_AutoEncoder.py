import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

encoding_dim = 8192
input_layer = Input(shape=(4096,))
encoded_layer = Dense(encoding_dim, activation='relu')(input_layer)
decoded_layer = Dense(4096, activation='sigmoid')(encoded_layer)

autoencoder = Model(input_layer, decoded_layer)
encoder = Model(input_layer, encoded_layer)
decoder_input = Input(shape=(encoding_dim,))
decoder = Model(decoder_input, autoencoder.layers[-1](decoder_input))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

faces_data = fetch_olivetti_faces(shuffle=True, random_state=42)
x_data = faces_data.images.astype('float32').reshape((len(faces_data.images), 4096))

split_idx = int(0.8 * len(x_data))
x_train, x_test = x_data[:split_idx], x_data[split_idx:]

autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

num_images = 10
plt.figure(figsize=(20, 4))
for i in range(num_images):
    plt.subplot(2, num_images, i + 1)
    plt.imshow(x_test[i].reshape(64, 64), cmap='gray')
    plt.axis("off")

    plt.subplot(2, num_images, i + 1 + num_images)
    reconstructed = autoencoder.predict(np.expand_dims(x_test[i], axis=0))
    plt.imshow(reconstructed.reshape(64, 64), cmap='gray')
    plt.axis("off")
plt.show()