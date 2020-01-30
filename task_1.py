"""Comparing neural networks trained using annealing and backpropagation"""
from copy import deepcopy
from math import exp
from random import random

import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Model
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm


def create_annealing_model(n_features, n_classes):
    input_layer = Input(shape=(n_features,))
    dense_1 = Dense(8, activation='relu')(input_layer)
    dense_2 = Dense(n_classes, activation='softmax')(dense_1)
    annealing_model = Model(inputs=input_layer, outputs=dense_2)
    annealing_model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
    return annealing_model


def extract_current_state(layers):
    """Extract network weights from layers"""
    return deepcopy([layer.get_weights() for layer in layers])


def initialize_weights(layers):
    """Initialize network weights with small random numbers"""
    for layer in layers:
        weights = np.array(layer.get_weights())
        new_weights = list()
        for weight_batch in weights:
            new_weights.append(
                np.random.normal(scale=0.1, size=weight_batch.shape))
        layer.set_weights(new_weights)


def new_weights_proposal(layers):
    """Provide new weights proposal for the network"""
    for layer in layers:
        weights = np.array(layer.get_weights())
        new_weights = list()
        for weight_batch in weights:
            new_weights.append(
                weight_batch +
                np.random.normal(scale=0.05, size=weight_batch.shape))
        layer.set_weights(new_weights)


def evaluate_performance(model, split_data):
    """Return the accuracy of the model on the testing set"""
    X_train, _, Y_train, _ = split_data
    loss, accuracy = model.evaluate(X_train, Y_train, verbose=0)
    return loss, accuracy


def restore_state(layers, weights):
    """Restore the state of network weights"""
    for layer, weight_batch in zip(layers, weights):
        layer.set_weights(weight_batch)


def simulated_annealing(split_data,
                        model,
                        layers,
                        max_iters: int,
                        initial_T: float = 2.0,
                        decay_constant: float = 1.0):
    """Use SA algorithm to train the neural network on Iris dataset

    Args:
        split_data: Tuple of X_train, X_test, Y_train, Y_test.
        model: Neural network model object.
        layers: Layers of the network.
        max_iters: Maximum iterations for the algorithm.
        initial_T: Starting temperature factor.
        decay_constant: lambda value in exponential decay formula.

    """
    initialize_weights(layers)
    best_loss, _ = evaluate_performance(model, split_data)
    current_loss = best_loss
    print('Starting loss', current_loss)
    current_state = extract_current_state(layers)
    best_weights = current_state

    T = initial_T
    for i in tqdm(range(max_iters)):
        new_weights_proposal(layers)
        new_loss, _ = evaluate_performance(model, split_data)
        if new_loss < current_loss:
            current_loss = new_loss
            if current_loss < best_loss:
                best_loss = current_loss
                best_weights = current_state
        else:
            try:
                acceptance_value = exp((current_loss - new_loss) / T)
            except ZeroDivisionError:
                break

            if random() < acceptance_value:
                current_loss = new_loss
                current_state = extract_current_state(layers)
            else:
                restore_state(layers, current_state)

        T = initial_T * exp(-decay_constant * (i + 1))

    return best_loss, best_weights


def test_backpropagation_learning(split_data, model, epochs):
    """Test the performance of the model when taught using backpropagation"""
    X_train, X_test, Y_train, Y_test = split_data
    model.fit(X_train,
              Y_train,
              epochs=epochs,
              verbose=0,
              validation_data=(X_test, Y_test))
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    return loss, accuracy


def main():
    """Driver method of the script"""
    iris = load_iris()
    data = iris['data']
    classes = iris['target']

    # One hot encoding
    encoder = OneHotEncoder()
    labels = encoder.fit_transform(classes[:, np.newaxis]).toarray()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled,
                                                        labels,
                                                        test_size=0.2,
                                                        random_state=42)

    n_features = data.shape[1]
    n_classes = labels.shape[1]

    annealing_model = create_annealing_model(n_features, n_classes)

    best_loss, _ = simulated_annealing(
        (X_train, X_test, Y_train, Y_test), annealing_model,
        annealing_model.layers[1:], 10000, 1, 1e-4)

    print('Best loss', best_loss)

    _, accuracy = annealing_model.evaluate(X_test, Y_test, verbose=0)
    print('Model accuracy', accuracy)

    backpropagation_model = create_annealing_model(n_features, n_classes)
    _, accuracy = test_backpropagation_learning(
        (X_train, X_test, Y_train, Y_test), backpropagation_model, 10000)
    print('Model accuracy', accuracy, '(backpropagation)')


if __name__ == '__main__':
    main()
