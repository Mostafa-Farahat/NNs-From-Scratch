# Nerual Network Report

## Methodology

- the code in `ReportGenerator.py` create 5 sets of hyperparameters and tests these 5 sets once while using the Sigmoid activation function and the other with the tanh activation function

- the report generates tables showing the results for the tests run using each activation function

- the test calculates an `overfit` metric which is the difference between the train accuracy and the test accuracy, the smaller this metric is the better. actually negative `overfit` values indicate that the model performed better on the test data than it has on the training data.

## Sigmoid Table
```
Config   Layers               LR       Epochs   Train %    Test %     Overfit   
----------------------------------------------------------------------
1        [5, 10, 3]           0.1      50       91.11      98.33      -7.22     
2        [5, 10, 3]           0.5      50       97.78      100.00     -2.22     
3        [5, 15, 8, 3]        0.1      50       86.67      83.33      3.33      
4        [5, 20, 10, 3]       0.3      100      98.89      100.00     -1.11     
5        [5, 8, 3]            0.2      75       97.78      100.00     -2.22     

----------------------------------------------------------------------
```

## Tanh Table
```
Config   Layers               LR       Epochs   Train %    Test %     Overfit   
----------------------------------------------------------------------
1        [5, 10, 3]           0.1      50       93.33      93.33      0.00      
2        [5, 10, 3]           0.5      50       64.44      63.33      1.11      
3        [5, 15, 8, 3]        0.1      50       100.00     96.67      3.33      
4        [5, 20, 10, 3]       0.3      100      100.00     96.67      3.33      
5        [5, 8, 3]            0.2      75       97.78      100.00     -2.22   
```

## Results

- As seen from the tables shown in the results section the best performance is obtained from sigmoid activation and tanh activation when the follwoing hyperparameters are used for each

|Activation Function| Train Acc. | Test Acc. | LR | Epochs | Layers | Hidden Nodes |
|:------------------|:----------:|:---------:|:--:|:------:|:------:|:------------:|
|    Sigmoid        |   97.78%   |    100%   | 0.2|   75   |  3     |    5, 8, 3   | 
|    Tanh           |   97.78%   |    100%   | 0.2|   75   |  3     |    5, 8, 3   | 

- It can be seen that for Sigmoid and Tanh the same set of hyperparameters are optimal and give identical results in terms of performance