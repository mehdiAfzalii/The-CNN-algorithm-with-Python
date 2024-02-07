import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from colorama import Fore

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_set = train_datagen.flow_from_directory('dataset/training_set',
                                              target_size=(64, 64),
                                              batch_size=32,
                                              class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

model.fit(train_set, epochs=25, validation_data=test_set)

model.save('cnn_model.h5')


def predict_image(model, img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        result = 'Dog'
    else:
        result = 'Cat'

    return result


def open_file():
    file_path = filedialog.askopenfilename()
    img = Image.open(file_path)
    img = img.resize((64, 64))
    img = ImageTk.PhotoImage(img)
    panel = tk.Label(root, image=img)
    panel.image = img
    panel.grid(row=2, column=1)
    predict(file_path)


def predict(img_path):
    prediction = predict_image(model, img_path)
    result_label.config(text=f'The image is predicted as a {prediction}')


root = tk.Tk()
root.title("Test")

button = tk.Button(root, text="Open Image", command=open_file)
button.grid(row=1, column=1)

result_label = tk.Label(root, text="")
result_label.grid(row=3, column=1)

model = tf.keras.models.load_model('cnn_model.h5')

root.mainloop()

print(Fore.GREEN + "\nCongratulations on the successful completion of the test!")
