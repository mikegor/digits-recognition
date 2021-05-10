import tensorflow as tf

def preprocess(file):
    audio_binary = tf.io.read_file(file)
    waveform, _ = tf.audio.decode_wav(audio_binary)
    waveform = tf.squeeze(waveform, axis=-1)
    zero_padding = tf.zeros([48000] - tf.shape(waveform), dtype=tf.float32)
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=130)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, -1)
    return spectrogram
