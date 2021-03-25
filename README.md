# Interpreting the Optical Absorption Spectra of Gold Nanoclusters by Machine Learning

UV-Vis absorption spectrometry has been widely utilized in identifying the compositions of metal nanoclusters (NCs) by comparing the experimental spectra with the reference data. However, the application of such method is limited when the optical absorption peaks are difficult to be identified, most of the time due to the sample being a mixture of many species. Herein, we develop a machine-learning-based method to interpret the compositions of metal NCs behind the spectra with a 1D Convolutional Neural Network (1D CNN). Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) models were used for comparisons with a fixed train-test split.

## Usage 
| Script | Description |
| ------------- | ------------------------------ |
| `uv_data_processing.py` | Data preprocessing for UV-Vis and composition data |
| `forward_hopt_full.py`, `reverse_hopt_full.py` | Hyperparameters tuning (Bayesian optimization with Gaussian process) for 1D CNN in forward and reverse predictions respectively |
| `forward_lstm_gru_hopt` | Hyperparameters tuning for LSTM and GRU models, argument: `LSTM` or `GRU`|
| `run_forward_final.py`, `run_reverse_final.py`| Model evaluations and predictions among 3 models and with training set size increment |
| `Demo / ` | In the works |

## Authors
Tiankai Chen, [Jiali Li](https://github.com/jiali1025), [Pengfei Cai](https://github.com/cpfpengfei)