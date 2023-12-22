# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
import seaborn as sns
# %%
# Veri yükleme
metadata = pd.read_csv("hamon.csv")
metadata.dropna(inplace=True)

# Veriyi yükleme ve işleme
df = pd.read_csv('hmnist_28_28_RGB.csv')
x = df.drop('label', axis=1).to_numpy() / 255
y = to_categorical(df['label'])

# Veri etiketlerini belirleme
label = {
    ' Actinic keratoses': 0,
    'Basal cell carcinoma': 1,
    'Benign keratosis-like lesions': 2,
    'Dermatofibroma': 3,
    'Melanocytic nevi': 4,
    'Melanoma': 5,
    'Vascular lesions': 6
}
# %%
x = x.reshape(-1, 28, 28, 3)

# Eğitim ve test verisi bölme
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.05, random_state=42)
# %%
# Veri artırma
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

data_generator.fit(train_x)

# Model oluşturma
model = Sequential()

model.add(Conv2D(64, (2, 2), input_shape=(28, 28, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(512, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Dropout(0.3))

model.add(Conv2D(1024, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Dropout(0.3))

model.add(Conv2D(1024, (1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(BatchNormalization())

model.add(Dropout(0.3))
model.add(Conv2D(1024, (1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(BatchNormalization())

model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=[top_k_categorical_accuracy])

model.summary()

# Eğitim
early = EarlyStopping(monitor='val_loss', patience=4, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, cooldown=0, mode='auto', min_delta=0.0001, min_lr=0)

class_weights = {0: 1, 1: 0.5, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
model.fit(train_x, train_y, epochs=50, batch_size=90, class_weight=class_weights, validation_data=(test_x, test_y), callbacks=[early, reduce_lr])

# Modeli kaydetme
model.save("model.h5")
# %%
# Modeli yükleme
model = load_model("model.h5")