import tensorflow as tf
from keras.datasets import fashion_mnist

from tensorflow.keras.models import Sequential

from tensorflow.keras import layers
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = fashion_mnist.load_data()

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)) / 255.0
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)) / 255.0

train_size = 60000
batch_size = 32
test_size = 10000

noise_factor = 0.2

x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape)
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)

x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0.0, clip_value_max=1.0)
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0.0, clip_value_max=1.0)


class DenoiseAutoEncoder(tf.keras.Model):
    def __init__(self):
        super(DenoiseAutoEncoder, self).__init__()

        self.encoder = Sequential(
            [
                layers.InputLayer(input_shape=(28, 28, 1)),
                layers.Conv2D(
                    16, kernel_size=3, strides=2, activation="relu", padding="same"
                ),
                layers.Conv2D(
                    8, kernel_size=3, strides=2, activation="relu", padding="same"
                ),
            ]
        )

        self.decoder = Sequential(
            [
                layers.Conv2DTranspose(
                    8, kernel_size=3, strides=2, padding="same", activation="relu"
                ),
                layers.Conv2DTranspose(
                    16, kernel_size=3, strides=2, padding="same", activation="relu"
                ),
                layers.Conv2DTranspose(1, kernel_size=3, strides=1, padding="same"),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)


model = DenoiseAutoEncoder()

optimizer = tf.keras.optimizers.Adam()

epochs = 10

model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())

model.fit(
    x_train_noisy,
    x_train,
    epochs=epochs,
    shuffle=True,
    validation_data=(x_test_noisy, x_test),
)

encoded_imgs = model.encoder(x_test).numpy()
decoded_imgs = model.decoder(encoded_imgs).numpy()

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.title("original + noise")
    plt.imshow(tf.squeeze(x_test_noisy[i]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    bx = plt.subplot(2, n, i + n + 1)
    plt.title("reconstructed")
    plt.imshow(tf.squeeze(decoded_imgs[i]))
    plt.gray()
    bx.get_xaxis().set_visible(False)
    bx.get_yaxis().set_visible(False)
plt.savefig("reconstruction.png")
plt.show()
