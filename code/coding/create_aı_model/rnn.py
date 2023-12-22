import tensorflow as tf
import time

def rnn(input_dim, output_dim, input_length, units,activation,use_bias,kernel_initializer,bias_initializer,dropout,out_activation,**kwargs):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=input_dim, output_dim=output_dim, embeddings_initializer=tf.keras.initializers.GlorotUniform(), input_length=input_length))
    
    model_sayısı = int(input("Kaç Tane Gizli Katman Yapmak İstersiniz: "))
    model_turu = input("Hangi RNN Türünü Kullanmak İstersiniz (SimpleRNN, GRU, LSTM): ")
    
    for _ in range(model_sayısı):
        if model_turu == "SimpleRNN":
            model.add(tf.keras.layers.SimpleRNN(units=units, activation=activation,use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, dropout=dropout,return_sequences=True))  # Örnek olarak 64 birim kullanıldı.
        elif model_turu == "GRU":
            model.add(tf.keras.layers.GRU(units=units,activation=activation,use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, dropout=dropout, return_sequences=True))  # Örnek olarak 64 birim kullanıldı.
        elif model_turu == "LSTM":
            model.add(tf.keras.layers.LSTM(units=units,activation=activation,use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, dropout=dropout, return_sequences=True))  # Örnek olarak 64 birim kullanıldı.
        else:
            print("Geçersiz RNN Türü! SimpleRNN, GRU veya LSTM seçmelisiniz.")

    time.sleep(2)
    print(f"{model_sayısı} Tane Gizli Katmanlı Modelin Oluşturuldu")
    model.add(tf.keras.layers.Dense(1, out_activation))
    print("""
        ANN Modelin Oluşturuldu
        """, model.summary())
    print("*"*20)
    save = input("Modelin Kaydedilmesini İster misin?(e/h): ").lower()
    if save == "e":
        model.save("my_model")
    else:
        print("İyi Günler")
        exit()
    
