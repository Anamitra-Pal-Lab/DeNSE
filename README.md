# DNN based State Estimator (DeNSE) for Transmission Systems
This repository contains the resources for implementation of the DNN based State Estimator in python programming language.

## Structure of Files
The python codes are divided into one main file named "Main_SE_EPRI.py" and other files containing the functions that are called in the main file.
The files containing the functions and their objectives are listed below.
1. Data loading - Loading the data in the order of inputs compatible with the DNN
2. Data pre processing.py - Pre processing functions making the techqniue robust for realistic data - incluldes noise addition and normalization
3. DNN training - Training the pre-processed data for a specific set of hyper parameters
4. Data post processing - Using the trained DNN model for evaluating the performance using the testing data (new real-time data), and visualization
5. Bad data detection and correction  - Demonstrating the DeNSE method's superiority in detecting and replacing bad data using the novel Nearest Operating Condition based BDDC



