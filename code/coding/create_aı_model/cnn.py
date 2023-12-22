import tensorflow as tf
import time

def cnn(filters, strides, activation, input_shape, pool_size, out_act):
    model = tf.keras.models.Sequential()
    katman_sayısı = int(input("Kaç Tane Conv2D Katman Oluşturmak İstersin: "))
    
    for _ in range(katman_sayısı):
        model.add(tf.keras.layers.Conv2D(filters, strides, activation, input_shape))
        
        model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size))
        
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, out_act))
    
    time.sleep(2)
    print(f"{katman_sayısı} Tane Gizli Katmanlı Modelin Oluşturuldu")
    print('''
        ANN Modelin Oluşturuldu
    ''')
    model.summary()
    print("*"*50)
    save = input("Modelin Kaydedilmesini İster misin?(e/h): ").lower()
    if save == "e":
        model.save("my_model.h5")
    else:
        print("İyi Günler")
        exit()
    
    

