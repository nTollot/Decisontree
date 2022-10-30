from tensorflow import keras
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Multiply
from keras.activations import softmax


class AuctionsAndChienModel(keras.Model):
    """
    Generic model for auctions and chien

    Parameters
    ----------
    units : unit class
        Number of neurons used per layer
    
    Returns
    ----------
    model : keras model
        Neural network model for auctions and chien
    """
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.hidden = Dense(units.hidden, activation="relu")
        self.hidden_value = Dense(units.hidden_value, activation="relu")
        self.hidden_policy = Dense(units.hidden_policy, activation="relu")
        self.value_output = Dense(1, activation="linear")
        self.policy_output = Dense(units.policy_output, activation="linear")

    def call(self, input):
        history_input, available_input = input
        hidden = self.hidden(history_input)

        hidden_policy = self.hidden_policy(hidden)
        policy_output = self.policy_output(hidden_policy)
        policy_output = Multiply()([policy_output, available_input])
        policy_output = softmax(policy_output)

        hidden_value = self.hidden_value(hidden)
        value_output = self.value_output(hidden_value)

        return policy_output, value_output


class MainModel(keras.Model):
    """
    Generic model for choosing the best card to play

    Parameters
    ----------
    units : unit class
        Number of neurons used per layer
    
    Returns
    ----------
    model : keras model
        Neural network model for choosing the best card to play
    """
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.embedding = Embedding(
            input_dim=units.input_dim, output_dim=units.output_dim, input_length=units.input_length)
        self.hidden = Dense(units.hidden, activation="relu")
        self.hidden_value = Dense(units.hidden_value, activation="relu")
        self.hidden_policy = Dense(units.hidden_policy, activation="relu")
        self.value_output = Dense(1, activation="linear")
        self.policy_output = Dense(units.policy_output, activation="linear")

    def call(self, input):
        history_input, available_input = input
        embedding = self.embedding(history_input)
        embedding = Flatten()(embedding)
        hidden = self.hidden(embedding)

        hidden_policy = self.hidden_policy(hidden)
        policy_output = self.policy_output(hidden_policy)
        policy_output = Multiply()([policy_output, available_input])
        policy_output = softmax(policy_output)

        hidden_value = self.hidden_value(hidden)
        value_output = self.value_output(hidden_value)

        return policy_output, value_output
