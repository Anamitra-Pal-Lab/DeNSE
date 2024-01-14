# DNN based State Estimator (DeNSE) for Transmission Systems
This repository contains the resources for implementation of the DNN based State Estimator in python programming language.

## Description of Files
The python codes are divided into one main file named "Main_SE_EPRI.py" and other files containing the functions that are called in the main file.
The files containing the functions and their objectives are listed below.
1. Data loading - Loading the data in the order of inputs compatible with the DNN
2. Data pre processing.py - Pre processing functions making the technique robust for realistic data - including noise addition and normalization
3. DNN training for state estimation - Training the pre-processed data for a specific set of hyperparameters
4. Data post processing - Using the trained DNN model for evaluating the performance using the testing data (new real-time data), and visualization
5. Bad data detection and correction (BDDC) - Demonstrating the DeNSE method's superiority in detecting and replacing bad data using the novel Nearest Operating Condition-based BDDC

## Sample Data for DeNSE
1. A sample data set for demonstrating the base topology of DeNSE can be found at https://www.dropbox.com/scl/fo/sayc5kd7t61mnkb4bhbml/h?rlkey=qneqvldwjxh57zv2sijbp7lqq&dl=0.
2. A sample data set for demonstrating the transfer learning capability of DeNSE for different topologies can be found at https://www.dropbox.com/scl/fo/unz106n0vhigokogbw4vj/h?rlkey=v7s0gdh2i7u300p10xebxkdet&dl=0.





