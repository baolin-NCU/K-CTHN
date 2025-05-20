# K-CTHN: Knowledge Embedded Convolutional-Transformer Hybrid Network 
This repository contains the code for the automatic modulation classification, we initially named the model Artificial Feature enhanced Convolutional Neural Network(AFECNN), so the folder is named like this.


## Installation
The code heavily depends on Tensorflow framework, which is 
optimized  for GPUs supporting CUDA. For our implementation the CUDA version 10.2 is used. Install the project
requirements with:
```
conda create -n afecnn python=3.8
conda activate afecnn
pip3 install -r requirements.txt
```

## Training Pipeline
We evaluated our approach on three datasets. [RML2016.10A](https://opendata.deepsig.io/datasets/2016.10/RML2016.10a.tar.bz), 
[RML2016.10C](https://opendata.deepsig.io/datasets/2016.04/2016.04C.multisnr.tar.bz)，[RML2016.10b]
Download them and extract them. By default, they are assumed to be in `/data/`

### Training
We use different optimization models for training. Various models can be found in the models folder.
Training and evaluation can be performed at the same time, and the results are stored in the runs folder.
```
python3 AFECNN/models/[model_name]/[model.py]
```


## Evaluation
We support automatic flops and runtime analysis for all  convolutional layers, the linear layer and 
batch normalization. As an example, to run an analysis of our model, you can use:
```
python3 AFECNN/evaluation/flops.py 
```
Inaddition, there is a separate flops calculation step in the training file, 

## Contributing
If you spot any bugs or if you are planning to contribute back bug-fixes, please open an issue and
discuss the feature with us.

