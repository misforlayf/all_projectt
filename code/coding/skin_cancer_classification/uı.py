import tkinter as tk
import random
import tensorflow as tf
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

model = load_model("model.h5")

metadata = pd.read_csv("hamon.csv")
metadata.dropna(inplace=True)

df = pd.read_csv('hmnist_28_28_RGB.csv')
x = df.drop('label', axis=1).to_numpy() / 255
y = to_categorical(df['label'])

label = {
    ' Actinic keratoses': 0,
    'Basal cell carcinoma': 1,
    'Benign keratosis-like lesions': 2,
    'Dermatofibroma': 3,
    'Melanocytic nevi': 4,
    'Melanoma': 5,
    'Vascular lesions': 6
}
x = x.reshape(-1, 28, 28, 3)


data_generator = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.10,
    height_shift_range=0.10,
    rescale=1/255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
data_generator.fit(x)

def predict_and_visualize():
    index = random.randint(0, len(x))
    test_image = x[index]

    predicted_probs = model.predict(np.expand_dims(test_image, axis=0))[0]
    predicted_label_index = np.argmax(predicted_probs)

    label_map = {
        0: 'Actinic keratoses',
        1: 'Basal cell carcinoma',
        2: 'Benign keratosis-like lesions',
        3: 'Dermatofibroma',
        4: 'Melanocytic nevi',
        5: 'Vascular lesions',
        6: 'Melanoma'
    }
    true_label_index = np.argmax(y[index])
    true_label = label_map[true_label_index]
    predicted_label = label_map[predicted_label_index]
    
    true_label_.config(text=f"Gerçek Sonuç: {true_label}")
    predict_label_.config(text=f"Tahmin Sonuç: {predicted_label}")

    plt.imshow(test_image)
    plt.title("Test Resmi")
    plt.show()

root = tk.Tk()
root.title("Skin Cancer Classification")
root.geometry("350x350")

random_button = tk.Button(root, text="Rastgele Resim", command=predict_and_visualize)
random_button.place(x=20, y=20)

true_label_ = tk.Label(root, text="Gerçek Sonuç: ")
true_label_.place(x=30, y=50)

predict_label_ = tk.Label(root, text="Tahmin Sonuç: ")
predict_label_.place(x=30, y=80)

root.mainloop()