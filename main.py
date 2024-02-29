from keras.models import model_from_json
from keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from category_encoders.one_hot import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.models import Sequential
import tensorflow as tf
from gpu_check import gpu_check


def load_and_prepare_data(url):
    dataset = pd.read_csv(url)
    dataset['price'] = dataset['price'].replace(
        {'\$': '', ',': ''}, regex=True).astype(float)
    dataset = dataset.dropna()

    le = LabelEncoder()
    colunas_categoricas = ['neighbourhood_cleansed',
                           'property_type', 'room_type']
    dataset[colunas_categoricas] = dataset[colunas_categoricas].apply(
        lambda col: le.fit_transform(col))

    scaler = StandardScaler()
    colunas_numericas = ['accommodates',
                         'bathrooms', 'bedrooms', 'beds', 'price']
    dataset[colunas_numericas + colunas_categoricas] = scaler.fit_transform(
        dataset[colunas_numericas + colunas_categoricas])

    return dataset


def build_model(input_shape, regularizer=None):
    model = Sequential([
        Dense(32, activation='relu', kernel_regularizer=regularizer,
              input_shape=input_shape),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    return model


def train_model(model, X_train, y_train, X_valid, y_valid, epochs=30):
    model.compile(loss='mean_squared_error', optimizer='sgd',
                  metrics=['mean_absolute_error'])
    history = model.fit(X_train, y_train, epochs=epochs,
                        verbose=1, validation_data=(X_valid, y_valid))
    return history


def evaluate_model(model, X_test, y_test):
    mse_test = model.evaluate(X_test, y_test)
    return mse_test


def save_model(model, model_json_path="model.json", model_weights_path="model.h5"):
    model_json = model.to_json()
    with open(model_json_path, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(model_weights_path)
    print("Saved model to disk")


def load_model(model_json_path="model.json", model_weights_path="model.h5"):
    json_file = open(model_json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_weights_path)
    print("Loaded model from disk")
    return loaded_model


# gpu_check()

url = "http://data.insideairbnb.com/united-states/ny/new-york-city/2024-02-06/data/listings.csv.gz"
dataset = load_and_prepare_data(url)

X = dataset.drop(columns=['price'])
y = dataset['price']

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=16)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=16)

model = build_model(input_shape=(X_train.shape[1],), regularizer=l2(l=0.01))
history = train_model(model, X_train, y_train, X_valid, y_valid)

evaluate_model(model, X_test, y_test)

save_model(model)
loaded_model = load_model()
