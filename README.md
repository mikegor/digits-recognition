# digits-recognition

It is project created by two-man team, modelled on tutorial from TensorFlow page. We used AudioMIST dataset(https://github.com/soerenab/AudioMNIST) which provides digits, spoken by 60 different people. 
This project works in following way:
We extract names of particular 'wav' files, shuffle them and group into train, val and test. Then we create waveforms with labels and print them to see results. No all the files have the same length so we add padding to each of them. Subsequently we created spectrograms by the use of Fourier transform, because they provide more valuable information about sound for neural network than waveforms. 

@ARTICLE{becker2018interpreting,
  author    = {Becker, S\"oren and Ackermann, Marcel and Lapuschkin, Sebastian and M\"uller, Klaus-Robert and Samek, Wojciech},
  title     = {Interpreting and Explaining Deep Neural Networks for Classification of Audio Signals},
  journal   = {CoRR},
  volume    = {abs/1807.03418},
  year      = {2018},
  archivePrefix = {arXiv},
  eprint    = {1807.03418},
}
