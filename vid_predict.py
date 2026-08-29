import argparse
import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Dropout, MaxPooling2D, Flatten, Dense

def build_fer_model(input_shape=(48, 48, 1), classes=7):
    num_features = 64
    model = Sequential()
    model.add(Input(shape=input_shape))

    # 1st stage
    model.add(Conv2D(num_features, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(num_features, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # 2nd stage
    model.add(Conv2D(num_features, (3, 3), activation='relu'))
    model.add(Conv2D(num_features, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 3rd stage
    model.add(Conv2D(2 * num_features, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(2 * num_features, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # 4th stage
    model.add(Conv2D(2 * num_features, (3, 3), activation='relu'))
    model.add(Conv2D(2 * num_features, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 5th stage
    model.add(Conv2D(4 * num_features, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(4 * num_features, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Flatten())

    # Fully connected neural networks
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(classes, activation='softmax'))
    return model

def main():
    ap = argparse.ArgumentParser(description="Facial Emotion Recognition on Video File")
    ap.add_argument('video', help='path to input video file')
    ap.add_argument('--output', help='path to output video file (optional)', default=None)
    args = vars(ap.parse_args())

    video_path = args['video']
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'", flush=True)
        sys.exit(1)

    weights_path = os.path.join('top_models', 'fer.h5')
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at '{weights_path}'", flush=True)
        sys.exit(1)

    cascade_path = 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        print(f"Error: Haar Cascade file not found at '{cascade_path}'", flush=True)
        sys.exit(1)

    print("Building model architecture and loading weights...", flush=True)
    model = build_fer_model()
    model.load_weights(weights_path)
    print("Model loaded successfully!", flush=True)

    face_haar_cascade = cv2.CascadeClassifier(cascade_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'", flush=True)
        sys.exit(1)

    emotions = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear']
    colors = {
        'Neutral': (200, 200, 200),
        'Happiness': (0, 230, 255),
        'Surprise': (255, 255, 0),
        'Sadness': (255, 100, 0),
        'Anger': (0, 0, 255),
        'Disgust': (0, 180, 0),
        'Fear': (180, 0, 180)
    }

    writer = None
    if args['output']:
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args['output'], fourcc, fps, (w, h))
        print(f"Recording output to '{args['output']}'", flush=True)

    window_name = 'Facial Emotion Recognition - Video'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 768)

    print("Processing video. Press 'q' or 'Esc' in video window to exit.", flush=True)

    while True:
        ret, img = cap.read()
        if not ret or img is None:
            break

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_detected = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=6)

        for (x, y, w, h) in faces_detected:
            roi_gray = gray_img[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = np.expand_dims(roi_gray, axis=-1)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels = img_pixels.astype('float32') / 255.0

            predictions = model.predict(img_pixels, verbose=0)
            max_index = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][max_index]) * 100

            predicted_emotion = emotions[max_index]
            box_color = colors.get(predicted_emotion, (0, 255, 0))

            cv2.rectangle(img, (x, y), (x + w, y + h), box_color, 2)

            label_text = f"{predicted_emotion} ({confidence:.1f}%)"
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            label_y = max(y - 10, text_h + 10)
            cv2.rectangle(img, (x, label_y - text_h - 5), (x + text_w + 10, label_y + baseline), box_color, cv2.FILLED)
            cv2.putText(img, label_text, (x + 5, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)

        if writer:
            writer.write(img)

        cv2.imshow(window_name, img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            break

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Finished video processing.", flush=True)

if __name__ == '__main__':
    main()
