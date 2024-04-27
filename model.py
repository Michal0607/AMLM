import numpy as np
from keras.layers import Dense,Dropout,LSTM,BatchNormalization
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras_tuner import RandomSearch, HyperParameters
import os

def train_data_generator(path, limit_speaker):
    files=os.listdir(path)
    i=0
    current_id=files[0].split("_")[0]

    for file in files:
        if current_id!=file.split("_")[0]:
            i+=1
            if i>=limit_speaker:
                break

        audio=np.load(os.path.join(path,file))
        current_id=file.split("_")[0]
        yield audio,current_id

def get_model():
    model=Sequential([
        LSTM(384,input_shape=(157,13)),
        Dense(256,activation='relu'),
        Dropout(0.25),
        Dense(96,activation='relu'),
        Dense(200,activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0003),loss='categorical_crossentropy',metrics=['accuracy'])

    return model

def history_plot(history):
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    accuracy=history.history['accuracy']
    val_accuracy=history.history['val_accuracy']
    
    sns.set(style="whitegrid")  
    
    plt.figure(figsize=(16,6))

    plt.subplot(1,2,1)
    sns.lineplot(x=range(1,len(loss)+1),y=loss,label='Train Loss',marker='o')  
    sns.lineplot(x=range(1,len(val_loss)+1),y=val_loss,label='Validation Loss',marker='o')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')

    plt.subplot(1,2,2)
    sns.lineplot(x=range(1,len(accuracy)+1),y=accuracy,label='Train Accuracy',marker='o')
    sns.lineplot(x=range(1,len(val_accuracy)+1),y=val_accuracy,label='Validation Accuracy',marker='o')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    
    plt.tight_layout()  
    plt.show()

if __name__ == "__main__":
    path = "C:/Users/48502/Desktop/Baza/IAD2SEM1/dataset5s_mfcc"
    generator = train_data_generator(path,200)
    X_train,y_train=[],[]
    for audio, speaker_id in generator:
        X_train.append(audio)
        y_train.append(speaker_id)

    X_train=np.array(X_train)
    y_train=np.array(y_train)
    encoder=LabelEncoder()
    y_train=encoder.fit_transform(y_train)
    num_speakers=len(encoder.classes_)
    y_train=keras.utils.to_categorical(y_train,num_classes=num_speakers)
    print(X_train.shape,y_train.shape)

    X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.2,random_state=42)

    model=get_model()
    model.summary()
    history=model.fit(X_train,y_train,epochs=25,batch_size=32,validation_data=(X_val,y_val))
    history_plot(history)
    model.save('AMLM/model.keras')