# Import TF and TF Hub libraries.
import tensorflow as tf
import tensorflow_hub as hub

import cv2
import time
import numpy as np
import argparse

# Download the model from TF Hub. (Pose Estimation)
model = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')
movenet = model.signatures['serving_default']

# Pose Classification
interpreter = tf.lite.Interpreter(model_path='Models/pose_classifier.tflite')
interpreter.allocate_tensors()

input_tensor = interpreter.get_input_details()[0]['index']
output_tensor = interpreter.get_output_details()[0]['index']

# Read Labels
with open('models/pose_labels.txt', "r") as file:
    # Read the entire file content
    file_content = file.read()
    labels = file_content.split('\n')

# Threshold for
threshold = .3


if __name__ == '__main__':
    par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
    par.add_argument('-C', '--camera', default=0, # required=True, # default=2,
                     help='Source of camera or video file path.')
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
        reshaped_array = keypoints[0][0].numpy().reshape((1, 51))
        interpreter.set_tensor(input_tensor, reshaped_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_tensor)
        output_label = max(output_data[0])
        output_label = output_data[0].tolist().index(output_label)
        print(labels[output_label], max(output_data[0]))

        img = cv2.putText(img, 'Class : %s  /  Score : %f' % (labels[output_label], max(output_data[0])), (10, 40), cv2.FONT_HERSHEY_COMPLEX,0.5, (0, 0, 0), 1)
        if not (time.time() - fps_time) == 0:
            img = cv2.putText(img, 'FPS: %f' % (1.0 / (time.time() - fps_time)),
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        fps_time = time.time()

        # Shows image
        cv2.imshow('frame', img)
        # Waits for the next frame, checks if q was pressed to quit
        if cv2.waitKey(1) == ord("q"):
            break

        # Reads next frame
        success, img = cap.read()

    cap.release()