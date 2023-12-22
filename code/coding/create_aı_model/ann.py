import tensorflow as tf
import time 

def ann(shape, units,activation, use_bias, kernel_initializer, bias_initializer, out_activation):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=shape))
    model_sayısı = int(input("Kaç Tane Gizli Katman Yapmak İstersiniz: "))

    for _ in range(model_sayısı):
        model.add(tf.keras.layers.Dense(units, activation, use_bias, kernel_initializer, bias_initializer))

    time.sleep(2)
    print(f"{model_sayısı} Tane Gizli Katmanlı Modelin Oluşturuldu")
    model.add(tf.keras.layers.Dense(1, out_activation))
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

