import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from PIL import Image  # Import the Image module to work with images
import os
from tensorflow.keras.callbacks import ModelCheckpoint  # Import the ModelCheckpoint callback

# Define a data generator
from tensorflow.keras.utils import Sequence


# class DataGenerator(Sequence):
#     def __init__(self, image_paths, ages, batch_size):
#         self.image_paths = image_paths
#         self.ages = ages
#         self.batch_size = batch_size
#
#     def __len__(self):
#         return int(np.ceil(len(self.image_paths) / self.batch_size))
#
#     def __getitem__(self, index):
#         start = index * self.batch_size
#         end = (index + 1) * self.batch_size
#         batch_x = self.image_paths[start:end]
#         batch_y = self.ages[start:end]
#         return self.__data_generation(batch_x, batch_y)
#
#     def __data_generation(self, batch_x, batch_y):
#         # Load and preprocess the images for the batch
#         image_data = []
#
#         for image_path in batch_x:
#             img = Image.open(image_path)
#             img = img.resize((200, 200))  # Resize the image to match the model's input shape
#             img = np.array(img) / 255.0  # Normalize the image
#             image_data.append(img)
#
#         image_data = np.array(image_data)
#
#         return image_data, batch_y


# Read your dataset, assuming it contains image paths ('image_path') and ages ('age')
xl = pd.ExcelFile("images and excel files/dataset_for_model.xlsx")

df = xl.parse("Лист1")
df = df[(df["age"] <= 40) & (df["age"] >= 3)]
# Load and preprocess ages
age_data = df['age'].values.reshape(-1, 1)
age_scaler = MinMaxScaler()
age_data = age_scaler.fit_transform(age_data)
#
# # Split the data into training, validation, and test sets
# X_train, X_temp, y_train, y_temp = train_test_split(df["image_path"], age_data, test_size=0.3, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
#
# # Create the data generators
# batch_size = 16  # Adjust this according to your available memory
# train_generator = DataGenerator(X_train, y_train, batch_size)
# validation_generator = DataGenerator(X_val, y_val, batch_size)
#
# # Load the pre-trained VGG-16 model (excluding the top layer)
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
#
# # Add custom layers for age prediction
# x = base_model.output
# x = keras.layers.GlobalAveragePooling2D()(x)
# x = keras.layers.Dense(1024, activation='relu')(x)
# predictions = keras.layers.Dense(1, activation='linear')(x)
#
# # Create the model
# model = Model(inputs=base_model.input, outputs=predictions)
#
# # Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
#
# # Create a new ModelCheckpoint callback
# model_checkpoint = ModelCheckpoint("face_analyze.h5", save_best_only=False)
#
# # Model training (continue from where it left off)
# model.fit(train_generator, epochs=10, validation_data=validation_generator, callbacks=[model_checkpoint])
#
# # Model evaluation
# loss, mae = model.evaluate(validation_generator)
# print(f"Mean Absolute Error on Validation Data: {mae}")

# #-------------------------------------------------------------------------------------------------------------------------
#
#
#
#
#



# for continue training model
#----------------------------------------------------------------------------------------------------------
# from tensorflow.keras.models import load_model
# #1/10,2/10
# model = load_model("face_analyze.h5")
# additional_epochs = 2
# model.fit(train_generator, epochs=2 + additional_epochs, initial_epoch=1, validation_data=validation_generator, callbacks=[model_checkpoint])
# loss, mae = model.evaluate(validation_generator)
# print(f"Mean Absolute Error on Validation Data: {mae}")
#----------------------------------------------------------------------------------------------------------


#for using model
def face_analyze(image_path):
    from tensorflow.keras.models import load_model
    model = load_model("face_analyze.h5")
    xl = pd.ExcelFile("images and excel files/dataset_for_model.xlsx")

    df = xl.parse("Лист1")
    df = df[(df["age"] <= 40) & (df["age"] >= 3)]
    # Load and preprocess ages
    age_data = df['age'].values.reshape(-1, 1)
    age_scaler = MinMaxScaler()
    age_data = age_scaler.fit_transform(age_data)
    img = Image.open(image_path)
    img = img.resize((200, 200))  # Resize the image to match the model's input shape
    img = np.array(img) / 255.0  # Normalize the image

    # Expand the dimensions to create a batch of size 1
    img = np.expand_dims(img, axis=0)

    # Make a prediction
    scaled_prediction = model.predict(img)
    predicted_age = age_scaler.inverse_transform(scaled_prediction)
    print(int(predicted_age[0][0]))
    return int(predicted_age[0][0])

# face_analyze("images and excel files/images/image1697267424.9461925.png")


























#-----------------------------------------------------------------------------------------------------------------------------
#
# # code for build an excel file for images
# # excel fil is ready
# xl = pd.ExcelFile("images and excel files/dataset_for_model.xlsx")
# df = xl.parse("Лист1")
# directory_path = 'images and excel files/images-2/'
# dictionary_for_excel_file = {"image_path":[], "age": []}
# file_names = os.listdir(directory_path)
# image_files = [file for file in file_names if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
# for i in image_files:
#     print(i)
#     dictionary_for_excel_file["image_path"].append(f"images and excel files/images-2/{i}")
#     dictionary_for_excel_file["age"].append(i.split('_')[0])
#
# print(dictionary_for_excel_file)
# df = pd.DataFrame(dictionary_for_excel_file,columns=["image_path","age"])
# with pd.ExcelWriter("images and excel files/dataset_for_model.xlsx", mode='a', if_sheet_exists='replace') as writer:
#     df.to_excel(writer, sheet_name="Лист1", index=False)
#
#
#
#
#


