import numpy as np
from tensorflow import keras
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle
import os

def prepare_data(path, limit):
    files = os.listdir(path)
    i = 0
    current_id = files[0].split("_")[0]
    for file in files:
        if current_id != file.split("_")[0]:
            i += 1
            if i >= limit:
                break
            current_id = file.split("_")[0]
        audio = np.load(os.path.join(path, file))
        yield audio, current_id

def LDA(X_train, y_train):
    RNNmodel = keras.models.load_model("C:/Users/48502/Desktop/Baza/IAD2SEM1/Projects/AMLM/model.keras")
    embedding_layer = keras.Model(inputs=RNNmodel.input, outputs=RNNmodel.layers[3].output)
    X_train_emb = []

    for x in X_train:
        train_emb = embedding_layer.predict(np.expand_dims(x, axis=0))
        X_train_emb.append(train_emb.flatten())  

    X_train_emb = np.array(X_train_emb)
    lda_model = LinearDiscriminantAnalysis(n_components=51)
    lda_model.fit(X_train_emb, y_train)

    with open("AMLM/LDA_model.pk", "wb") as file:
        pickle.dump(lda_model, file)

if __name__ == "__main__":
    path = "C:/Users/48502/Desktop/Baza/IAD2SEM1/dataset5s_mfcc"
    generator = prepare_data(path, 200)
    X_train = []
    y_train = []
    for audio, current_id in generator:
        X_train.append(audio)
        y_train.append(current_id)
    LDA(X_train, y_train)
