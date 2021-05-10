import sounddevice as sd
import time
import numpy as np

from scipy.io.wavfile import write

fs = 48000  # Sample rate
seconds = 0.9  # Duration of recording

i = 0
name = input('write your name and press enter...')
while i<10:
    input('press enter and say: ' + str(i))
    time.sleep(0.3)
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()  # Wait until recording is finished
    write('output/'+str(i) + '_' + name +'.wav', fs, myrecording)  # Save as WAV file 
    i = i + 1