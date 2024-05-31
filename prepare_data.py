import numpy as np
import os
import librosa 

def read_audio_files(path,path_out,fol_out,time):
    path_out=os.path.join(path_out,fol_out)
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    files_list,id_list,audio_combined=[],[],[]
    folders_list=os.listdir(path)

    for folders in folders_list:
        folders=os.path.join(path,folders)
        speakers_list=os.listdir(folders)
        id_list.append(folders.split(os.path.sep)[-1])

        for folder in speakers_list:
            folder=os.path.join(folders,folder)

            for file in os.listdir(folder):
                if file.endswith('.flac'):
                    file=os.path.join(folder,file)
                    files_list.append(file)
    id_set=set(id_list)

    id=files_list[0].split(os.path.sep)[-3]
    for file in files_list:
        file_id=file.split(os.path.sep)[-3]
        if file_id in id_set and id==file_id:
            audio,sr=librosa.load(file,sr=16000)
            audio_combined.append(audio)
            id=file_id
        elif id!=file_id:
            speakers={
                "id":id,
                "audio":np.concatenate(audio_combined)
            }
            audio_combined=[]
            id=file_id
            build_dataset(speakers,path_out,time)

def build_dataset(speaker_dict,path_out,time):
    sample_rate=16000
    id=speaker_dict['id']
    sample_length=sample_rate*time
    num_samples=len(speaker_dict['audio'])//sample_length

    for i in range(num_samples):
        start=i*sample_length
        end=start+sample_length
        sample=speaker_dict['audio'][start:end]
        sample_mfcc=librosa.feature.mfcc(y=sample,sr=sample_rate,n_mfcc=13)
        sample_mfcc=sample_mfcc.T

        output_file=os.path.join(path_out,f'{id}_{i+1}.npy')
        np.save(output_file,sample_mfcc)


if __name__=="__main__":
    path="C:/Users/48502/Desktop/Baza/IAD2SEM1/Projects/LibriSpeech/train-clean-100"
    path_output="C:/Users/48502/Desktop/Baza/IAD2SEM1"
    folder_out="dataset5s_mfcc"
    time=5
    
    read_audio_files(path,path_output,folder_out,time)