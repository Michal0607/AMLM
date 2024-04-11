import numpy as np
from keras.layers import Dense,Dropout,LSTM,Sequential,BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import librosa
import os

def train_data(path,limit_speaker,limit_files=1000):
    X_train,y_train=[],[]
    files=os.listdir(path)
    i,j=0,0
    while i<limit_speaker:
        id=files[0].split("_")[0]
        for file in files:
            if id!=file.split("_")[0] or j>=limit_files:
                if id==file.split("_")[0]:
                    continue
                i+=1
                j=0
            audio=np.load(os.path.join(path,file))
            id=file.split("_")[0]
            X_train.append(audio)
            y_train.append(id)
            j+=1

    return np.array(X_train),np.array(y_train)

def model():
    model=Sequential(
        LSTM(160,input_shape=(157,13),return_sequences=True),
        LSTM(128),
        Dense(64),
        Dense(50)
    )
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    return model

if __name__=="__main__":
    path="C:/Users/48502/Desktop/Baza/IAD2SEM1/dataset5s_mfcc"
    X_train,y_train=train_data(path,200,150)