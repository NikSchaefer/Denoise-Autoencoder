# Denoise Autoencoder

Autoencoder to denoise images based on the fashion_mnist dataset

## How it works

The noisy data is randomly applied to the data with a noise factor of 0.2 and then clipped between 0.0 and 1.0. 

The encoder comprises of 2 Convolutional layers used to highlight the features and negate the noise. The decoder works in sequence and uses Conv2dTranspose layers to undo the Convolutional layers but without the noise.

## Data

Data is from the fashion_mnist dataset on [tensorflow datasets](https://github.com/tensorflow/datasets)

```py
from keras.datasets import fashion_mnist
(x_train, _), (x_test, _) = fashion_mnist.load_data()
```

## Installation

Install Python 3.8+

```bash
pip install tensorflow keras numpy
```

## License

[MIT](https://choosealicense.com/licenses/mit/)
