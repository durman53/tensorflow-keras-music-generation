#module mido for working with midi files
from mido import MidiFile, Message
import os
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import *

#load midi file
data = []
pattern = MidiFile("musics/Hot N Cold - Chorus.mid")
a = []
#get all messages from pattern
for msg in pattern.tracks[2]:
	#if it message not meta message:
        if msg.type == 'note_on' or msg.type == 'note_off':
	    #get message type("note_on" or "note_off"), note and velocity to m variable
            m = msg.bytes()
	    #preprocessing data
            if m[0] == 144:m[0]=1
            else:m[0]=0
            m[1] = m[1]/127
            m[2] = m[2]/127
	    #append preprocessed time of message
            m.append(msg.time/96/6)
            a.append(m)
        if len(a) == 101:
            data.append(a[:-1])
            break
#in out midi file already all meta messages, open it
out = MidiFile('base.mid')
dat = np.array(data, dtype='float32')
#load pretrained model
model = load_model('model.h5')

#adding preprocessed data to out midi file
for i, n in enumerate(dat[0]):
    if n[0] == 1:ty = 144
    else:ty=128
    vel = n[2]*127
    arr = [ty, round(n[1]*127), round(vel)]
    msg = Message.from_bytes(arr, round(n[3]*96*6))
    out.tracks[2].insert(i+19, msg)

i = 0
#generate music by lenght 2500 messages in a loop
while len(out.tracks[2]) < 2500:
    print(f'{len(out.tracks[2])}-messages of 2500')
    pred = model.predict(dat)
    
    #postprocess model output
    if np.argmax(pred[0]) == 1:ty = 144
    else:ty = 128
    arr = [ty, np.argmax(pred[1]), np.argmax(pred[2])]
    #generate message from postprocessed data
    msg = Message.from_bytes(arr, round(np.argmax(pred[3])*96))
    out.tracks[2].insert(119+i, msg)
    m = msg.bytes()
    #append model output to data and delete 1st element of data
    if m[0] == 144:m[0]=1
    else:m[0]=0
    m[1] = m[1]/127
    m[2] = m[2]/127
    m.append(round(msg.time/96/6))
    data[0] = data[0][1:]
    data[0].append(m)
    dat = np.array(data, dtype='float32')
    i+=1
#save output music
out.save('test.mid')
