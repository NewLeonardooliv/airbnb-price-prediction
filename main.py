from keras.models import model_from_json
from keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from category_encoders.one_hot import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.models import Sequential
import tensorflow as tf
from gpu_check import checkUsingGPU


checkUsingGPU()

url = "http://data.insideairbnb.com/united-states/ny/new-york-city/2024-02-06/data/listings.csv.gz"
dataset = pd.read_csv(url)
dataset['price'] = dataset['price'].replace(
    {'\$': '', ',': ''}, regex=True).astype(float)

dataset = dataset[["neighbourhood_cleansed", "property_type",
                   "room_type", "accommodates", "bathrooms", "bedrooms", "beds", "price"]]

dataset.info(verbose=True)
dataset.select_dtypes(include='object').describe()
dataset.select_dtypes(include='float').describe()
dataset.select_dtypes(include='int').describe()
dataset = dataset.dropna()


le = LabelEncoder()

colunas_categoricas = ['neighbourhood_cleansed', 'property_type', 'room_type']
dataset[colunas_categoricas] = dataset[colunas_categoricas].apply(
    lambda col: le.fit_transform(col))

colunas_numericas = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 'price']

scaler = StandardScaler()
dataset[colunas_numericas + colunas_categoricas] = scaler.fit_transform(
    dataset[colunas_numericas + colunas_categoricas])

print(dataset)

col = ['neighbourhood_cleansed', 'property_type', 'room_type',
       'accommodates', 'bathrooms', 'bedrooms', 'beds', 'price']

dataset = pd.DataFrame(dataset, columns=col)

print(dataset)

X = dataset.drop(columns=['price'])
y = dataset['price']

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=16
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.25,
    random_state=16
)

regularizer_l1 = l1(l=0.01)
regularizer_l2 = l2(l=0.012)
regularizer_combined = l1_l2(l1=0.01, l2=0.01)

# Model
model = Sequential()
model.add(Dense(32, activation='relu',
          kernel_regularizer=regularizer_l2, input_shape=X_train.shape[1:]))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.summary()


model.compile(loss='mean_squared_error', optimizer='sgd',
              metrics=['mean_absolute_error'])

history = model.fit(X_train, y_train,
                    epochs=30,
                    verbose=1,
                    validation_data=(X_valid, y_valid)
                    )

X_new = X_test[:3]

y_pred = model.predict(X_new)

print(y_pred)

num_epochs = 30

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, num_epochs),
         history.history["loss"], label="Training Loss")
plt.plot(np.arange(0, num_epochs),
         history.history["mean_absolute_error"], label="Training Mean Absolute Error")
plt.plot(np.arange(0, num_epochs),
         history.history["val_loss"], label="Validation Loss")
plt.plot(np.arange(0, num_epochs),
         history.history["val_mean_absolute_error"], label="Validation Mean Absolute Error")
plt.title("Training and Validation Metrics")
plt.xlabel("Epoch #")
plt.ylabel("Metrics")
plt.legend()
plt.show()

mse_test = model.evaluate(X_test, y_test)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")
print("Loaded model from disk")
