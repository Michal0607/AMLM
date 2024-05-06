import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

path="C:/Users/48502/Desktop/Baza/IAD2SEM1/Projects/AMLM/lda_lstm_results_1.csv"

data=pd.read_csv(path)

impostor_test=data[data['Test']==0]
genuine_test=data[data['Test']==1]
num_bins=60

hist_impostor,bins_impostor=np.histogram(impostor_test['Score'],bins=num_bins,range=(0, 1))
far_values = []
for i in range(1,len(bins_impostor)):
    far_values.append(len(impostor_test[impostor_test['Score']>=bins_impostor[i]])/len(impostor_test))

hist_genuine, bins_genuine = np.histogram(genuine_test['Score'],bins=num_bins,range=(0, 1))
frr_values = []
for i in range(1,len(bins_genuine)):
    frr_values.append(len(genuine_test[genuine_test['Score']<bins_genuine[i]])/len(genuine_test))

x_values = np.linspace(0,1,num_bins)

plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.plot(x_values,far_values,label='FAR',color='red')
plt.plot(x_values,frr_values,label='FRR',color='blue')
plt.xlabel('Score')
plt.ylabel('Value')
plt.title('FAR-FRR Curves')
plt.xlim(0,1)
plt.legend()
plt.grid()

eer_index=np.argmin(np.abs(np.array(far_values)-np.array(frr_values)))
eer=far_values[eer_index]

far_percent=np.array(far_values)*100
frr_percent=np.array(frr_values)*100

plt.subplot(2, 1, 2)
plt.xscale('log')  
plt.yscale('log')  
plt.xlim(0.5, 100)
plt.ylim(0.5, 100)
plt.xticks([1,5,10,25,100],['1%','5%','10%','25%','100%'])
plt.yticks([1,5,10,25,100],['1%','5%','10%','25%','100%'])
plt.plot(far_percent, frr_percent, color='purple')
plt.xlabel("False Acceptance Rate (FAR in %)")
plt.ylabel("False Rejection Rate (FRR in %)")
plt.title('DET Curve')

plt.plot(far_percent[eer_index],frr_percent[eer_index], 'ro')
plt.annotate(f'EER={eer * 100:.2f}%',(far_percent[eer_index] + 0.5,frr_percent[eer_index]))
plt.grid()

plt.tight_layout()
plt.show()
