# DNN based State Estimator (DeNSE) for Transmission Systems
This repository contains the resources for implementation of the DNN based State Estimator. This is based on the paper ["Time-Synchronized Full System State Estimation Considering Practical Implementation Challenges"](https://arxiv.org/abs/2212.01729) which is currently under review in the Journal of Modern Power Systems and Clean Energy.  

## Motivation
Power utilities commonly prioritize the placement of Phasor Measurement Units (PMUs) at higher voltage levels before considering lower voltage levels. This strategy results in a significant portion of lower voltage levels in transmission systems remaining unmonitored. Traditional state estimation methods require complete system observability to accurately estimate states, a condition not always met in these scenarios. To address this gap, we introduce a novel learning-based method, DeNSE, designed specifically to accurately estimate the states of lower voltage levels in transmission systems, even in situations where full observability is not achieved.
## Languages and Dependencies
The entire program was implemented using Python programming language.  Specifically, the open-source Python package Tensorflow was used to train the neural network models. For the data generation, the Matpower library in Matlab was used. 

The standard Python libraries like Numpy, Pandas, Matplotlib etc. need to be installed to run these codes. For detailed installation instructions for Windows, please follow these [instructions](https://www.tensorflow.org/install/pip#windows). 

## Description of Files
The python codes are divided into one main file named "Main_SE_EPRI.py" and other files containing the functions that are called in the main file.
The files containing the functions and their objectives are listed below.
1. Data loading - Loading the data in the order of inputs compatible with the DNN
2. Data pre processing.py - Pre processing functions making the technique robust for realistic data - including noise addition and normalization
3. DNN training for state estimation - Training the pre-processed data for a specific set of hyperparameters
4. Data post processing - Using the trained DNN model for evaluating the performance using the testing data (new real-time data), and visualization
5. Bad data detection and correction (BDDC) - Demonstrating the DeNSE method's superiority in detecting and replacing bad data using the novel Nearest Operating Condition-based BDDC

## Sample Data for DeNSE
1. A sample data set for demonstrating the DeNSE functionalities on the base topology can be found at https://www.dropbox.com/scl/fo/sayc5kd7t61mnkb4bhbml/h?rlkey=qneqvldwjxh57zv2sijbp7lqq&dl=0.
2. A sample data set for training the transfer learning capability of DeNSE for different topologies can be found at https://www.dropbox.com/scl/fo/unz106n0vhigokogbw4vj/h?rlkey=v7s0gdh2i7u300p10xebxkdet&dl=0.
3. A sample data set for demonstrating the transfer learning capability of DeNSE for different topologies can be found at https://www.dropbox.com/scl/fo/1esngxtp19g4hx6z490j3/h?rlkey=2desjrgbaohubj2a8mth8iez3&dl=0

## Sample Simulation Results
1. Performance of DeNSE under different noise models for IEEE 118-bus system

2. Transfer learning results

## Usage of Files
Download all the files provided in this repository and store them in a folder. Download the sample data set provided above and keep it in the same folder. Run the main file ('Main_EPRI_SE.py') to train the DNN, and evaluate the state estimation performance.



