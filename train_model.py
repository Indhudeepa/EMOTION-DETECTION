import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

print("Loading dataset...")
data = pd.read_csv("fer2013.csv")

pixels = data['pixels'].tolist()

images = []
for pixel_sequence in pixels:
    image = np.array(pixel_sequence.split(), dtype='float32')
    image = image.reshape(48, 48, 1)
    images.append(image)

images = np.array(images)
images = images / 255.0

labels = to_categorical(data['emotion'], num_classes=7)

X_train = images[data['Usage'] == 'Training']
y_train = labels[data['Usage'] == 'Training']

X_test = images[data['Usage'] != 'Training']
y_test = labels[data['Usage'] != 'Training']

print("Building model...")

model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Training started...")
model.fit(X_train, y_train, epochs=10, batch_size=64)

print("Evaluating model...")
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)

model.save("emotion_model.h5")

print("Model saved successfully!")
