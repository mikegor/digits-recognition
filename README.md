# digits-recognition

It is project created by two-man team, modelled on tutorial from TensorFlow page. We used AudioMIST dataset(https://github.com/soerenab/AudioMNIST) which provides digits, spoken by 60 different people.<br/><br/>
This project works in following way:<br/>
We extract names of particular 'wav' files, shuffle them and group into train, val and test. Then we create waveforms with labels and print them to see results. No all the files have the same length so we add padding to each of them. Subsequently we create spectrograms by the use of Fourier transform, because they provide more valuable information about sound for neural network than waveforms. Now the data is ready to use in neural network<br/>
We use convoluntional neural network with two convoluntional layers, resizing layer to simplify input, normalization layer to improve efficiency and dropout layer to prevent overfitting. Then with Adam optimizer and SparseCategoricalCrossentropy loss function we train model through 10 epochs. Afterwards we test model on previously excluded data and data from external source.<br/>
We can see that the score on audioMNIST data is almost perfect. On the other hand, when we use data that we have recorded with our friends the score is about 50%. It means that the model is overfitted(needing more data) but still taking into consideration the simplicity of the model and size of dataset the score is quite good.<br/><br/>
We also created some tools that allows us to test our model in live. These are programs to record audio samples proper to our model or live speech recognizer using our model. They are in the branch live-predict.<br/>  

audioMNIST citation:
<br/>
@ARTICLE{becker2018interpreting,
  author    = {Becker, S\"oren and Ackermann, Marcel and Lapuschkin, Sebastian and M\"uller, Klaus-Robert and Samek, Wojciech},
  title     = {Interpreting and Explaining Deep Neural Networks for Classification of Audio Signals},
  journal   = {CoRR},
  volume    = {abs/1807.03418},
  year      = {2018},
  archivePrefix = {arXiv},
  eprint    = {1807.03418},
}
