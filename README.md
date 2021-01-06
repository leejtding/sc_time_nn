# sc_time_nn
This repository consists of a linear model for classifying scRNA-seq expressions
by time point and an RNN for generating scRNA-seq expressions at the last time point
in a given dataset.

The linear model can be run via linear_main.py and the RNN model can be run via
rnn_main.py.

The example data contained in data.zip consists of scRNA-seq data for 271 human
skeletal muscle myoblasts collected 0, 24, 48, and 72 hours after the switch of
human myoblast culture from growth to differentiation media. The paper describing
the dataset can be found via the following link: https://www.nature.com/articles/nbt.2859. 