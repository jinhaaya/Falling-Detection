# Import TF and TF Hub libraries.
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

import cv2
import time
import numpy as np
import argparse


# Download the model from TF Hub. (Pose Estimation)
model = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')
movenet = model.signatures['serving_default']

# Pose Classification - Load the saved model
loaded_model = keras.models.load_model("your_model.h5")

# Read Labels
with open('models/pose_labels.txt', "r") as file:
    # Read the entire file content
    file_content = file.read()
    labels = file_content.split('\n')

# Threshold for
threshold = .3


if __name__ == '__main__':
    par = argparse.ArgumentParser(description='Human Fall Detection.')
    par.add_argument('-C', '--camera', default=0, # required=True,
                     help='Source of camera or video file path.')
    par.add_argument('-V', '--visualize', default=False, # required=True,
                     help='Visualize the output.')
    args = par.parse_args()

    # Loads video source (0 is for main webcam)
    video_source = args.camera
    cap = cv2.VideoCapture(video_source)

    # Checks errors while opening the Video Capture
    if not cap.isOpened():
        print('Error loading video')
        quit()

    success, img = cap.read()

    if not success:
        print('Error reding frame')
        quit()

    y, x, _ = img.shape

    fps_time = time.time()
    f = 1
    temp_array = []

    output_label = 1
    output = [[0, 0]]
    while success:
        # A frame of video or an image, represented as an int32 tensor of shape: 256x256x3. Channels order: RGB with values in [0, 255].
        tf_img = cv2.resize(img, (192, 192))
        tf_img = cv2.cvtColor(tf_img, cv2.COLOR_BGR2RGB)
        tf_img = np.asarray(tf_img)
        tf_img = np.expand_dims(tf_img, axis=0)

        # Resize and pad the image to keep the aspect ratio and fit the expected size.
        image = tf.cast(tf_img, dtype=tf.int32)

        # Run model inference.
        outputs = movenet(image)
        # Output is a [1, 1, 17, 3] tensor.
        keypoints = outputs['output_0']

        # iterate through keypoints
        for k in keypoints[0, 0, :, :]:
            # Converts to numpy array
            k = k.numpy()

            # Checks confidence for keypoint
            if k[2] > threshold:
                # The first two channels of the last dimension represents the yx coordinates (normalized to image frame, i.e. range in [0.0, 1.0]) of the 17 keypoints
                yc = int(k[0] * y)
                xc = int(k[1] * x)

                # Draws a circle on the image for each keypoint
                img = cv2.circle(img, (xc, yc), 2, (0, 255, 0), 5)


        # Pose Classification
        temp_array.append(keypoints[0][0].numpy())
        if len(temp_array) == 30:
            temp_array = temp_array[-20:]
            temp_np_array = np.array([temp_array])
            output = loaded_model.predict(temp_np_array,steps=1,verbose=0)
            print(output[0])
            output_label = max(output[0])
            output_label = output[0].tolist().index(output_label)

        if args.visualize:
            if output_label == 0:
                img = cv2.putText(img, 'Class : %s  /  Score : %.2f' % (labels[output_label], max(output[0])), (10, 40),
                                  cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
            elif output_label == 1:
                img = cv2.putText(img, 'Class : %s  /  Score : %.2f' % (labels[output_label], max(output[0])), (10, 40),
                                  cv2.FONT_HERSHEY_COMPLEX,0.5, (0, 0, 0), 1)
            if not (time.time() - fps_time) == 0:
                img = cv2.putText(img, 'FPS: %f' % (1.0 / (time.time() - fps_time)),
                                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            # Shows image
            cv2.imshow('frame', img)
        else:
            print("--------------------------------")
            print(f"Result : {labels[output_label]}")
            print(f"Score : {max(output[0])}")
            print(f"FPS : {1.0 / (time.time() - fps_time)}")

        fps_time = time.time()
        # Waits for the next frame, checks if q was pressed to quit
        if cv2.waitKey(1) == ord("q"):
            break

        # Reads next frame
        success, img = cap.read()

    cap.release()