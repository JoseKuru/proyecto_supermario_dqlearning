import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Este sera nuestro modelo
class ConvNet(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(ConvNet, self).__init__()
        # Tendremos 3 capas convolucionales
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            # Aplanamos y creamos una capa adiccional con 512 neuronas
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        # Definimos la función de propagación
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


# Mismo modelo con Keras
def build_model(input_shape, n_actions):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=32, strides=4, activation='ReLU', input_shape=input_shape))
    model.add(keras.layers.Conv2D(32, kernel_size=64, strides=2, activation='ReLU'))
    model.add(keras.layers.Conv2D(32, kernel_size=64, strides=1, activation='ReLU'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='ReLU'))
    model.add(keras.layers.Conv2D(n_actions, activation='linear'))

    return model