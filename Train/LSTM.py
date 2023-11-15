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
import argparse

par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
par.add_argument('-L', '--load', default=False,  # required=True, # default=2,
                 help='Use saved NumPy data.')
par.add_argument('-S', '--save', default=False,  # required=True, # default=2,
                 help='save NumPy data.')
args = par.parse_args()

if not args.load :
    # Download the model from TF Hub. (Pose Estimation)
    movenet = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')
    movenet = movenet.signatures['serving_default']

    # Generate some example data (replace this with your actual data)
    import numpy as np


    # Define Train, Val, Test set
    X = []
    y = []
    X_fall_train = []
    y_fall_train = []
    X_fall_val = []
    y_fall_val = []
    X_fall_test = []
    y_fall_test = []

    # falling_video
    fall_videos = []
    fall_filepath = ['dataset/fall/'+i for i in os.listdir('dataset/fall')]
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
            fps = cap.get(cv2.CAP_PROP_FPS)
            if frame_count % (fps//20) == 0 and len(output_frames) < 20:
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
        if len(output_frames) != 20 : continue
        output_frames = np.array(output_frames)
        X.append(output_frames)
        y.append([1, 0]) # idx0 for falling, idx 1 for non-falling

    print(np.array(X).shape)
    X_fall_train, X_temp, y_fall_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_fall_val, X_fall_test, y_fall_val, y_fall_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


    # falling_image
    fall_videos = []
    fall_filepath = ['dataset/fall_image/' + i for i in os.listdir('dataset/fall_image/')]

    temp = []
    for dir in tqdm(fall_filepath, desc="Data Processing for Falling Images"):
        images = os.listdir(dir)
        output_frames = []  # List to store selected frames
        for image in images:
            image = cv2.imread(dir + '/' + image)
            frame = cv2.resize(image, (192, 192))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frame = np.expand_dims(frame, axis=0)

            # Resize and pad the image to keep the aspect ratio and fit the expected size.
            frame = tf.cast(frame, dtype=tf.int32)
            output = movenet(frame)['output_0']
            output_frames.append(output[0][0])
        cap.release()

        # Convert the list of frames to a NumPy array
        if len(output_frames) != 20 : continue
        output_frames = np.array(output_frames)
        X.append(output_frames)
        y.append([1, 0]) # idx0 for falling, idx 1 for non-falling

    print(np.array(X).shape)
    X_fallimage_train, X_temp, y_fallimage_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_fallimage_val, X_fallimage_test, y_fallimage_val, y_fallimage_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


    # non-falling
    X = []
    y = []

    non_fall_videos = []
    non_fall_filepath = ['dataset/non_fall/UCF-101/'+i for i in os.listdir('dataset/non_fall/UCF-101')]
    # non_fall_filepath = ['dataset/non_fall/temp/'+i for i in os.listdir('dataset/non_fall/temp')]
    for files in non_fall_filepath :
        temp = os.listdir(files)
        for t in temp:
            filename = files+'/'+t
            non_fall_videos.append(filename)

    temp = []
    for video in tqdm(non_fall_videos, desc="Data Processing for non-Falling Videos"):
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
            if frame_count % 20 == 0 and len(output_frames) < 20:
                frame = cv2.resize(frame, (192, 192))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = np.asarray(frame)
                frame = np.expand_dims(frame, axis=0)

                # Resize and pad the image to keep the aspect ratio and fit the expected size.
                frame = tf.cast(frame, dtype=tf.int32)
                output = movenet(frame)['output_0']
                output_frames.append(output[0][0])
        cap.release()
        if len(output_frames) != 20 : continue

        # Convert the list of frames to a NumPy array
        # output_frames = np.array(output_frames)
        # print(output_frames.shape)
        X.append(output_frames)
        y.append([0, 1]) # idx0 for falling, idx 1 for non-falling

    print(np.array(X).shape)
    X_nonfall_train, X_temp, y_nonfall_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_nonfall_val, X_nonfall_test, y_nonfall_val, y_nonfall_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)



    X_fall_train = np.array(X_fall_train);    X_fallimage_train = np.array(X_fallimage_train);    X_nonfall_train = np.array(X_nonfall_train);
    y_fall_train = np.array(y_fall_train);    y_fallimage_train = np.array(y_fallimage_train);    y_nonfall_train = np.array(y_nonfall_train);
    X_fall_val = np.array(X_fall_val);    X_fallimage_val = np.array(X_fallimage_val);    X_nonfall_val = np.array(X_nonfall_val);
    y_fall_val = np.array(y_fall_val);    y_fallimage_val = np.array(y_fallimage_val);    y_nonfall_val = np.array(y_nonfall_val);
    X_fall_test = np.array(X_fall_test);    X_fallimage_test = np.array(X_fallimage_test);    X_nonfall_test = np.array(X_nonfall_test);
    y_fall_test = np.array(y_fall_test);    y_fallimage_test = np.array(y_fallimage_test);    y_nonfall_test = np.array(y_nonfall_test);

    X_train = np.append(X_fall_train, X_nonfall_train, axis=0);     X_train = np.append(X_train, X_fallimage_train, axis=0)
    y_train = np.append(y_fall_train, y_nonfall_train, axis=0);     y_train = np.append(y_train, y_fallimage_train, axis=0)
    X_val = np.append(X_fall_val, X_nonfall_val, axis=0);    X_val = np.append(X_val, X_fallimage_val, axis=0)
    y_val = np.append(y_fall_val, y_nonfall_val, axis=0);    y_val = np.append(y_val, y_fallimage_val, axis=0)
    X_test = np.append(X_fall_test, X_nonfall_test, axis=0);    X_test = np.append(X_test, X_fallimage_test, axis=0)
    y_test = np.append(y_fall_test, y_nonfall_test, axis=0);    y_test = np.append(y_test, y_fallimage_test, axis=0)

    print(X_train.shape)

    # X_train = np.array(X_fall_train)
    # y_train = np.array(y_fall_train)
    # X_val = np.array(X_fall_val)
    # y_val = np.array(y_fall_val)
    # X_test = np.array(X_fall_test)
    # y_test = np.array(y_fall_test)

    if args.save:
        np.save('dataset/X_train', X_train)
        np.save('dataset/y_train', y_train)
        np.save('dataset/X_val', X_val)
        np.save('dataset/y_val', y_val)
        np.save('dataset/X_test', X_test)
        np.save('dataset/y_test', y_test)
    print("Train Set :", len(X_fall_train)+len(X_fallimage_train), len(X_nonfall_train))
    print("Validation Set :", len(X_fall_val)+len(X_fallimage_val), len(X_nonfall_val))
    print("Test Set :", len(X_fall_test)+len(X_fallimage_test), len(X_nonfall_test))

    print("Data Processing Finished")

else:
    print("Use Saved NumPy dataset")
    X_train = np.load('dataset/X_train.npy', allow_pickle=True)
    y_train = np.load('dataset/y_train.npy', allow_pickle=True)
    X_val = np.load('dataset/X_val.npy', allow_pickle=True)
    y_val = np.load('dataset/y_val.npy', allow_pickle=True)
    X_test = np.load('dataset/X_test.npy', allow_pickle=True)
    y_test = np.load('dataset/y_test.npy', allow_pickle=True)

    print("Train Set :", len(X_train))
    print("Validation Set :", len(X_val))
    print("Test Set :", len(X_test))


# X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
# y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
# X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
# y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
# X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
# y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

print("Train Start")

# Define your LSTM model
model = keras.Sequential()
model.add(layers.Reshape((20, 51), input_shape=(20, 17, 3)))  # Choose the desired dimension as 'time_steps'
model.add(layers.LSTM(128, return_sequences=True))
model.add(layers.LSTM(128, return_sequences=False))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


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