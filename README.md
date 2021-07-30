# tensorflow-keras-music-generation
<h1>This is tensorflow-keras based Reccurent Neural Networks(RNN) model for music generation.</h1>

<h2>Dataset:</h2>
Dataset is midi piano examples of random musics. All dataset files locate on musics folder.

<h2>What is midi files?</h2>
Midi(Musical Instrument Digital Interface) keeps data of music - this is notes, notes time, velocity, channel.

<h2>How to run:</h2>
Project requirements: tensorflow>2.0, numpy, mido.
You can try generate music with running generate.py script.
Try to train net with rinning train.py script.

<h2>Model structure:</h2>
______________________________________________________________________________<br>
Layer (type)                    Output Shape         Param #     Connected to <br>                 
==============================================================================<br>
input_1 (InputLayer)            [(None, 100, 4)]     0                        <br>                    
______________________________________________________________________________<br>
lstm (LSTM)                     (None, 100, 128)     68096       input_1[0][0]<br>
______________________________________________________________________________<br>
lstm_1 (LSTM)                   (None, 128)          131584      lstm[0][0]   <br>                    
______________________________________________________________________________<br>
dense (Dense)                   (None, 128)          16512       lstm_1[0][0] <br>                    
______________________________________________________________________________<br>
dropout (Dropout)               (None, 128)          0           dense[0][0]  <br>                    
______________________________________________________________________________<br>
type (Dense)                    (None, 2)            258         dropout[0][0]<br>                    
______________________________________________________________________________<br>
note (Dense)                    (None, 128)          16512       dropout[0][0]<br>                    
______________________________________________________________________________<br>
vel (Dense)                     (None, 128)          16512       dropout[0][0]<br>                    
______________________________________________________________________________<br>
time (Dense)                    (None, 6)            774         dropout[0][0]<br>                   
==============================================================================<br>

<h2>Checkpoints:</h2>
When net train it save weights in any epoch to checkpoints folder

<h2>Example of generated music:</h2>


https://user-images.githubusercontent.com/76785927/127591443-204209c1-2c42-401a-8b45-080cf5b4c3c8.mp4


