Project from Advanced Machine Learning Methods

The main goal of the project is to create a biometric system that will recognize speakers and prevent false verification.
A collection of recordings from librispeech was used in the project training set of 100 hours "clean" speech.

### Project Files Description

1. **`prepare_data.py`**:
   - Processes the LibriSpeech dataset, transforming 100 hours of "clean" speech into 5-second MFCC audio segments. Organizes data for training and evaluation.
   
2. **`model.py`**:
   - Defines and trains the speaker verification model using a dataset of 200 speakers. Details the model architecture and training process.
   
3. **`LDA_transform.py`**:
   - Implements LDA to enhance feature separability, reducing dimensionality for improved discrimination in speaker verification.
   
4. **`evaluate.py`**:
   - Evaluates the trained model using a test set from the remaining 51 speakers. Outputs performance metrics.
   
5. **`metrics.py`**:
   - Computes key metrics like FAR, FRR, and possibly other relevant statistics for assessing the biometric system's effectiveness.
   
6. **`plot_results.py`**:
   - Generates visual results including DET curves and EER plots, under different test scenarios for clear visual interpretation of the model's performance.

### Evaluation Scenarios

The system was evaluated under three scenarios:
1. Scenario 1: 40-second enrollment sample, 10-second test sample.
2. Scenario 2: 60-second enrollment sample, 20-second test sample.
3. Scenario 3: 50-second enrollment sample, 30-second test sample.

Models were trained using Random Search to find the best parameters, including an LDA model to enhance evaluation results. Plots for FAR, FRR, and DET curves were generated to indicate EER, providing a comprehensive assessment of the system's reliability and accuracy.

Author: Michał Szulierz