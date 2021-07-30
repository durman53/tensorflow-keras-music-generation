#module mido for working with midi files
from mido import MidiFile, Message
import os
import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint

data = []
target = []
target1 = []
target2 = []
target3 = []

#listing music dir and open all tracks
for file in os.listdir('musics'):
    #reading midi files
    pattern = MidiFile('musics/'+file)
    a = []
    b = []
    c = []
    d = []
    e = []
    #get all messages of track
    for msg in pattern.tracks[2]:
        #if message is not meta message
        if msg.type == 'note_on' or msg.type == 'note_off':
            #reading message data and preprocessing
            m = msg.bytes()
            if m[0] == 144:m[0]=1
            else:m[0]=0
            m[1] = m[1]/127
            m[2] = m[2]/127
            m.append(msg.time/96/6)
            a.append(m)
            b.append(m[0])
            c.append(round(m[1]*127))
            d.append(round(m[2]*127))
            e.append(round(m[3]*6))
        #if len of data is 101 append it to data variable and delete 1st element
        if len(a) == 101:
            data.append(a[:-1])
            target.append(b[100])
            target1.append(c[100])
            target2.append(d[100])
            target3.append(e[100])
            a = a[1:]
            b = b[1:]
            c = c[1:]
            d = d[1:]
            e = e[1:]

#convert lists to numpy arrays
data = np.array(data)
target = np.array(target, dtype='float32')
target1 = np.array(target1)
target2 = np.array(target2)
target3 = np.array(target3)
#neural network initializing with input shape (100, 4)
inp = Input((100, 4))
#hidden layers
hidden = LSTM(128, return_sequences=True)(inp)
hidden = LSTM(128)(hidden)
hidden = Dense(128)(hidden)
hidden = Dropout(0.3)(hidden)
#4 outputs
out1 = Dense(2, activation='softmax', name='type')(hidden)
out2 = Dense(128, activation='softmax', name='note')(hidden)
out3 = Dense(128, activation='softmax', name='vel')(hidden)
out4 = Dense(6, activation='softmax', name='time')(hidden)

model = Model(inputs=[inp], outputs=[out1, out2, out3, out4])
#compile net
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
#make net checkpoints
filepath = "checkpoints/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(
    filepath, monitor='loss', 
    verbose=0,     
    save_best_only=True,        
    mode='min')
#removing all old checkpoits
for i in os.listdir('checkpoints'):
    os.remove('checkpoints/'+i)
#train net with 200 epochs
model.fit(data, [target, target1, target2, target3],
          epochs=200, batch_size=64, validation_split=0,
          verbose=2, callbacks=[checkpoint])
#save model, but we can use checkpoints
model.save('model.h5')
