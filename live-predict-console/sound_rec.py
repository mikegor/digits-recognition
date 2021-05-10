import sounddevice as sd
import tensorflow as tf
from scipy.io import wavfile
from scipy.io.wavfile import write
import func
import numpy as np
import sys
import time

fs = 48000  # Sample rate
seconds = 0.9  # Duration of recording

model = tf.keras.models.load_model('digits_model.h5')

while 1:
    input('press enter to record')
    time.sleep(0.3)
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()  # Wait until recording is finished
    write('output.wav', fs, myrecording)  # Save as WAV file 
    data = func1.preprocess('output.wav')
    test_audio = []
    test_audio.append(data.numpy())
    test_audio = np.array(test_audio)
    y_pred = np.argmax(model.predict(test_audio), axis=1)
    print('prediction:', y_pred)