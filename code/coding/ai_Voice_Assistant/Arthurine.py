import speech_recognition as sr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
import os
from gtts import gTTS
from playsound import playsound
import random

# Modeli yükleyin
model = load_model("model1.h5")

# Veriyi yükleyin ve etiketleri düzeltin
df = pd.read_csv("data.csv").head(100000)

# Tokenizer'ı oluşturun ve veriyi işleyin
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["Review"].astype(str).str.lower())

# Tüm kelime sayısını alın
total_words = len(tokenizer.word_index) + 1

# Tokenize edilmiş dizileri oluşturun
tokenized_sequences = tokenizer.texts_to_sequences(df["Review"].astype(str))

# Giriş dizilerini oluşturun
input_sequences = []
for i in tokenized_sequences:
    for t in range(1, len(i)):
        n_gram_sequence = i[:t + 1]
        input_sequences.append(n_gram_sequence)

# max_sequence_len = max([len(x) for x in input_sequences])
os.system("cls")

# Kayıt işlemini gerçekleştiren fonksiyonu tanımlayın
def record():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Bir şeyler söyleyin...")
        audio = r.listen(source)
        try:
            voice = r.recognize_google(audio, language='tr-TR')
            print("Söylenen: " + voice)
            return voice
        except sr.UnknownValueError:
            print("Anlaşılamadı")
            return ""
        except sr.RequestError:
            print("Sistem hatası")
            return ""

def speak(string):
    tts = gTTS(string, lang='tr')
    rand = random.randint(1,10000)
    file = 'audio-'+str(rand)+'.mp3'
    tts.save(file)
    playsound(file)
    os.remove(file)

# Sohbetbotunu düzeltilmiş versiyonunu tanımlayın
def Arthurine(seedtext, nextwords):
    for _ in range(nextwords):
        token_list = tokenizer.texts_to_sequences([seedtext])[0]
        token_list = pad_sequences([token_list], maxlen=108, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)[0]
        predicted_index = np.argmax(predicted_probs)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        seedtext += " " + output_word
    return seedtext

# Konuşma tanıma ve sohbetbotunu kullanma işlemini başlatın
if __name__ == "__main__":
    while True:
        voice = record()
        if voice:
            # speak(voice)
            response = Arthurine(voice, 30)
            print("Arthurine: " + response)
            speak(response)
        elif voice in "kapat" or voice in "çıkış":
                print("Arthurine: Görüşürüz.")
                exit()
