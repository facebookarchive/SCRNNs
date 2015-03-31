# Structurally Constrained Recurrent Neural Network

This is a self contained software accompanying the paper titled: Learning
Longer Memory in Recurrent Neural Networks: http://arxiv.org/abs/1412.7753.
The code allows you to reproduce our results on two language modeling datasets:
* PenntreeBank
* Text8

The code implements three recurrent models:
* Standard Recurrent Neural Networks
* Long Short Term Memory Recurrent Neural Networks
* Structurally Constrained Recurrent Neural Networks

It also allows you to play around with various hyper-parameters.

## Examples
Here are some of the examples of how to use the code.

* To run a standard RNN model on PenntreeBank with following
hyper-parameters:
  * hidden units: 100
  * minibatch size: 32
  * learning rate: 0.05

you type
```
cuth -i main.lua -dset ptb -name srnn_sm -nhid 100 -batchsz 32 -eta 0.05
```

* To run a LSTM RNN model on Text8 with following
hyper-parameters:
  * hidden units: 100
  * minibatch size: 32
  * learning rate: 0.05
  * unfolding depth: 20
  * backprop frequency: 5

you type
```
cuth -i main.lua -dset text8 -name lstm_sm -nhid 100 -batchsz 32 -eta 0.05 -blen 20 -bfreq 5
```

* To run a Structurally Constrained RNN model on PenntreeBank with following
hyper-parameters:
  * hidden units: 100
  * number of constrained units: 20
  * minibatch size: 32
  * learning rate: 0.05
  * unfolding depth: 30
  * backprop frequency: 5

you type
```
cuth -i main.lua -dset text8 -name scrnn_sm -nhid 100 -nslow 20 -batchsz 32 -eta 0.05 -blen 30 -bfreq 5
```

To list all the options available, you need to type
```
cuth main.lua --help
```


## Requirements
The software requires you to have the following two packages already
installed on your systems:
* Torch 7
* fbcunn
It runs on standard Linux box.


## Installing
Download the files in an appropriate directory and run
the code from there. See below.


## How Structurally Constrained Recurrent Neural Network Software works
The top level file is called main.lua. In order to run the code
you need to run the file using torch. For example:
```
cuth -i main.lua -<option1_name> option1_val -<option2_name> option2_val ...
```

In order to check what all options are available, type
```
cuth -i main.lua --help
```

## License
Structurally Constrained Recurrent Neural Network is BSD-licensed.
We also provide an additional patent grant.


## Other Details
See the CONTRIBUTING file for how to help out.
