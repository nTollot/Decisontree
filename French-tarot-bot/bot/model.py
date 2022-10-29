from tensorflow import keras
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Model
from keras.layers import Multiply
from keras.activations import softmax


class AuctionsAndChienModel(keras.Model):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = Dense(units.hidden1,activation="relu")
        self.hidden2 = Dense(units.hidden2,activation="relu")
        self.hidden3 = Dense(units.hidden3,activation="relu")
        self.hidden4 = Dense(units.hidden4,activation="relu")
        self.value_output = Dense(units.policy_input,activation="linear")

    def call(self, input):
        hidden1 = self.hidden1(input)
        hidden2 = self.hidden2(hidden1)
        hidden3 = self.hidden3(hidden1)
        output_value = self.










