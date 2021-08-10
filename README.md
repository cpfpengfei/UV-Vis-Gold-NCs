# Decoding Complexity in Chemical Compositions from Optical Absorption Spectra by Machine Learning

Rapid and accurate chemical composition identification is critically important in chemistry. While it can be achieved with optical absorption spectrometry by comparing the experimental spectra with the reference data when the chemical compositions are simple, such application is limited in more complicated scenarios. This is due to the difficulties in identifying optical absorption peaks (i.e., from featureless spectra) arose from the complexity. In this work, using the UV-Vis absorption spectra of metal nanoclusters (NCs) as a demonstration, we develop a machine-learning-based method to unravel the compositions of metal NCs behind the featureless spectra. By implementing a one-dimensional Convolutional Neural Network (CNN), good matches between prediction results and experimental results and low mean absolute error values are achieved on these optical absorption spectra that human cannot interpret. This work opens a door for the identification of nanomaterials at molecular precision from their optical properties, paving the way to rapid and high-throughput characterizations.

## Code Availability 
The code for 1D CNN, Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) models are in this repository. These models were compared on a fixed train-test split.

| Script | Description |
| ------------- | ------------------------------ |
| `uv_data_processing.py` | Data preprocessing for UV-Vis and composition data |
| `forward_hopt_full.py`, `reverse_hopt_full.py` | Hyperparameters tuning (Bayesian optimization with Gaussian process) for 1D CNN in forward and reverse predictions respectively |
| `forward_lstm_gru_hopt.py`, `reverse_lstm_gru_hopt.py` | Hyperparameters tuning for LSTM and GRU models, argument: `LSTM` or `GRU`|
| `run_forward_final.py`, `run_reverse_final.py`| Model evaluations and predictions among 3 models and with training set size increment |

## Authors
Tiankai Chen, [Jiali Li](https://github.com/jiali1025), [Pengfei Cai](https://github.com/cpfpengfei)
