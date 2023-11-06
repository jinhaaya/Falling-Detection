import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split

import os
import cv2
import time
import numpy as np
import argparse
from tqdm import tqdm

# Download the model from TF Hub. (Pose Estimation)
movenet = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')
movenet = movenet.signatures['serving_default']


# Define your LSTM model
model = keras.Sequential()
model.add(layers.Reshape((20, 51), input_shape=(20, 17, 3)))  # Choose the desired dimension as 'time_steps'
model.add(layers.LSTM(128, return_sequences=True))
model.add(layers.LSTM(128, return_sequences=False))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Generate some example data (replace this with your actual data)
import numpy as np


# Define Train, Val, Test set
X = []
y = []
X_train = []
y_train = []
X_val = []
y_val = []
X_test = []
y_test = []

# falling
fall_videos = []
fall_filepath = ['dataset/fall/'+f'{i}' for i in range(1, 23)]
for files in fall_filepath :
    temp = os.listdir(files)
    for t in temp:
        filename = files+'/'+t
        fall_videos.append(filename)

temp = []
for video in tqdm(fall_videos, desc="Data Processing for Falling Videos"):
    cap = cv2.VideoCapture(video)
    # Initialize variables
    frame_count = 0  # Count of frames read
    output_frames = []  # List to store selected frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if we have reached the end of the video
        frame_count += 1
        # Save every 10th frame
        if frame_count % 15 == 0 and len(output_frames) < 20:
            frame = cv2.resize(frame, (192, 192))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frame = np.expand_dims(frame, axis=0)

            # Resize and pad the image to keep the aspect ratio and fit the expected size.
            frame = tf.cast(frame, dtype=tf.int32)
            output = movenet(frame)['output_0']
            output_frames.append(output[0][0])
    cap.release()

    # Convert the list of frames to a NumPy array
    output_frames = np.array(output_frames)
    # print(output_frames.shape)
    X.append(output_frames)
    y.append([1, 0]) # idx0 for falling, idx 1 for non-falling

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# non-falling


print("Data Processing Finished")
print("Train Start")


X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Train the model
model.fit(
    X_train, y_train,  # Training data
    epochs=80,        # Adjust the number of epochs based on your dataset
    batch_size=16,    # Batch size
    validation_data=(X_val, y_val),  # Validation data
    verbose=1
)

print("Train Finished")
print("Test Start")
# '''
# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)  # Replace with your test data
print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")
# '''

print("Test Finished and model saved")
model.save("your_model.h5")