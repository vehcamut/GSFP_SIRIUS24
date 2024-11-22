import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

class PricingNetwork:
    def __init__(self, train, test):
        '''
        Parameters
        ----------
        train : tuple
            Inputs and outputs in train set
        test : tuple
            Inputs and outputs in test set
        '''
        self.X_train, self.y_train = train
        self.X_test, self.y_test = test

        self.normalizer = layers.Normalization(input_shape=[self.X_train.shape[1],])
        self.normalizer.adapt(self.X_train)

    def create_ann(self):
        self.pricing_model = keras.Sequential([
            self.normalizer,
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(1, activation='relu')
        ])

        self.pricing_model.compile(loss='mean_squared_error',
                                   optimizer=tf.keras.optimizers.Adam())

    def fit_model(self, epochs=100):
        self.history = self.pricing_model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=epochs
        )
    
