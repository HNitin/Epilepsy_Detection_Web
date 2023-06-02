import os
import pyedflib
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, LSTM, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def analyze_edf_file(edf_file_path):
    # Open the EDF file
    edf_file = pyedflib.EdfReader(edf_file_path)

    # Get the number of signals/channels in the EDF file
    num_signals = edf_file.signals_in_file

    # Read all the signal samples
    signal_samples = []
    for signal_index in range(num_signals):
        samples = edf_file.readSignal(signal_index)
        signal_samples.append(samples)

    # Convert signal_samples to a numpy array
    signal_samples = np.array(signal_samples)

    # Transpose the signal samples to have channels as the last dimension
    signal_samples = np.transpose(signal_samples)

    # Close the EDF file
    edf_file.close()

    # Perform feature scaling
    scaler = StandardScaler()
    signal_samples = scaler.fit_transform(signal_samples)

    # Reshape the data for LSTM input [samples, timesteps, features]
    signal_samples = np.reshape(signal_samples, (signal_samples.shape[0], 1, signal_samples.shape[1]))

    return signal_samples

def perform_epilepsy_detection(model_path, edf_file_path):
    # Load the saved model
    autoencoder = load_model(model_path)

    # Analyze the EDF file and obtain the signal samples
    signal_samples = analyze_edf_file(edf_file_path)

    # Use the trained autoencoder to reconstruct the signal samples
    reconstructed_samples = autoencoder.predict(signal_samples)

    # Calculate the reconstruction error
    reconstruction_error = np.mean(np.square(signal_samples - reconstructed_samples))

    # Perform epilepsy detection based on the reconstruction error
    if reconstruction_error > 0.5:
        result = "Epilepsy detected in file: " + os.path.basename(edf_file_path)
    else:
        result = "No epilepsy detected in file: " + os.path.basename(edf_file_path)

    return result

"""
def calculate_accuracy(model_path, test_file_path):
    # Load the saved model
    autoencoder = load_model(model_path)

    # Analyze the EDF file and obtain the signal samples
    signal_samples = analyze_edf_file(test_file_path)

    # Use the trained autoencoder to reconstruct the signal samples
    reconstructed_samples = autoencoder.predict(signal_samples)

    # Calculate the reconstruction error
    reconstruction_error = np.mean(np.square(signal_samples - reconstructed_samples))

    # Calculate accuracy
    accuracy = 1.0 - reconstruction_error

    return accuracy
"""
