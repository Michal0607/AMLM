import numpy as np
import pandas as pd 
import keras
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import load_model
import os

def test_data_generator(path,limit):
    list_temp = []
    current_speaker, speakers_count, check = 0, 0, 0

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".npy"):
                file_path = os.path.join(root, file)
                filename = os.path.splitext(file)[0]
                speaker_id = int(filename.split('_')[0])

                if speaker_id != current_speaker:
                    current_speaker = speaker_id
                    speakers_count += 1
                    check = 0

                mfcc_matrix = np.load(file_path)
                if mfcc_matrix.shape[0] == 157:
                    if speakers_count > limit:
                        list_temp.append(mfcc_matrix) # 8 12 10
                        if len(list_temp) == 10:
                            X_enrollment = np.concatenate(list_temp, axis=0)
                            y_enrollment = speaker_id
                            list_temp = []
                            check = 1
                            yield X_enrollment, y_enrollment, 'enrollment'
                        if len(list_temp) == 6 and check == 1: # 2 4 6
                            X_test = np.concatenate(list_temp, axis=0)
                            y_test = speaker_id
                            list_temp = []
                            yield X_test, y_test, 'test'

def genuine_impost_test(test_sample,embedding_layer,mean_enrollment_embedding,j,i,test,scores):
        test_embeddings=[]
        test_samples_157=[test_sample[k:k+157] for k in range(0,len(test_sample),157)]
        for test_sample_157 in test_samples_157:
            test_embedding=embedding_layer.predict(np.expand_dims(test_sample_157,axis=0))[0,:]
            test_embeddings.append(test_embedding)

        mean_test_embedding=np.expand_dims(np.mean(test_embeddings,axis=0),axis=0)
        similarity_score=cosine_similarity(mean_enrollment_embedding,mean_test_embedding)
        is_genuine=test
        scores.append([y_enrollment[j],y_test[i],similarity_score[0][0],is_genuine])

def score(embedding_layer,X_enrollment,y_enrollment,X_test,y_test,output_csv):
    scores=[]
    print(type(scores))
    for j in range(len(X_enrollment)):
        enrollment_embeddings =[]
        enrollment_sample = X_enrollment[j]
        enrollment_samples_157 = [enrollment_sample[k:k+157] for k in range(0, len(enrollment_sample), 157)]
        for enrollment_sample_157 in enrollment_samples_157:
            enrollment_embedding = embedding_layer.predict(np.expand_dims(enrollment_sample_157, axis=0))[0,:]
            enrollment_embeddings.append(enrollment_embedding)
        mean_enrollment_embedding = np.expand_dims(np.mean(enrollment_embeddings, axis=0),axis=0)

        for i in range(len(X_test)):
            test_sample = X_test[i]
            random=round(np.random.uniform(0,1),2)
            
            if y_enrollment[j]==y_test[i] and random<=0.7:
                genuine_impost_test(test_sample,embedding_layer,mean_enrollment_embedding,j,i,1,scores)
            elif y_enrollment[j]!=y_test[i] and random<=0.12:
                genuine_impost_test(test_sample,embedding_layer,mean_enrollment_embedding,j,i,0,scores)
        print(f"Próbka numer: {j}")
    df = pd.DataFrame(scores,columns=['Speaker A','Speaker B','Score','Test'])
    df.to_csv(output_csv,index=False)

if __name__ == '__main__':
    path = "C:/Users/48502/Desktop/Baza/IAD2SEM1/dataset5s_mfcc"
    limit = 200
    X_enrollment,X_test,y_enrollment,y_test = [],[],[],[]

    data_generator = test_data_generator(path, limit)

    for X, y, dataset_type in data_generator:
        if dataset_type == 'enrollment':
            X_enrollment.append(X)
            y_enrollment.append(y)
        elif dataset_type == 'test':
            X_test.append(X)
            y_test.append(y)
    model=load_model('AMLM/model.keras')
    model.summary()
    embedding_layer=keras.Model(inputs=model.layers[0].input,outputs=model.layers[3].output)
    score(embedding_layer,X_enrollment,y_enrollment,X_test,y_test,'AMLM/results_3.csv')