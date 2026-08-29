import cv2
import argparse
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
    ap = argparse.ArgumentParser(description="Facial Emotion Recognition on Image File")
    ap.add_argument('image', help='path to input image file')
    args = vars(ap.parse_args())

    image_path = args['image']
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'", flush=True)
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

    classifier = cv2.CascadeClassifier(cascade_path)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not decode image file '{image_path}'", flush=True)
        sys.exit(1)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_detected = classifier.detectMultiScale(gray_img, scaleFactor=1.18, minNeighbors=5)

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

    if len(faces_detected) == 0:
        print("No faces detected in the provided image.", flush=True)
    else:
        print(f"Detected {len(faces_detected)} face(s).", flush=True)

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

        # Draw bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h), box_color, 2)

        # Label box
        label_text = f"{predicted_emotion} ({confidence:.1f}%)"
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        label_y = max(y - 10, text_h + 10)
        cv2.rectangle(img, (x, label_y - text_h - 5), (x + text_w + 10, label_y + baseline), box_color, cv2.FILLED)
        cv2.putText(img, label_text, (x + 5, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)

        print(f"Face at ({x}, {y}, {w}, {h}): Predicted {predicted_emotion} with {confidence:.1f}% confidence", flush=True)

    # Save output image alongside input or display
    output_path = os.path.splitext(image_path)[0] + "_prediction.png"
    cv2.imwrite(output_path, img)
    print(f"Saved prediction image to: {output_path}", flush=True)

    window_name = 'Facial Emotion Recognition'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 768)
    cv2.imshow(window_name, img)
    print("Press any key in the image window to close.", flush=True)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
