import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def FAR_FRR_plot(path):
    data_frame = pd.read_csv(path)

    is_genuine_0 = data_frame[data_frame['Test'] == 0]
    is_genuine_1 = data_frame[data_frame['Test'] == 1]

    sns.set(style="whitegrid")

    plt.subplot(2, 1, 1)
    sns.histplot(is_genuine_0['Score'], bins=30, kde=True, color='red', label='is_genuine = 0')
    plt.xlabel('Wartość score')
    plt.ylabel('Liczba wystąpień')
    plt.title('Histogram testu impostor')
    plt.legend(loc='upper right')

    plt.subplot(2, 1, 2)
    sns.histplot(is_genuine_1['Score'], bins=30, kde=True, color='blue', label='is_genuine = 1')
    plt.xlabel('Wartość score')
    plt.ylabel('Liczba wystąpień')
    plt.title('Histogram testu genuine')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

path = "C:/Users/48502/Desktop/Baza/IAD2SEM1/Projects/AMLM/lda_lstm_results_3.csv"