<div align="center">
<p align="left">
  <img src="imgs/Logos/title.png" >
</p>

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.10-3776AB?&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.12-EE4C2C?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Pytorch Lightning 1.7-792EE5?logo=pytorchlightning&logoColor=white"></a>

<a href="https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.loggers.mlflow.html#mlflow"><img alt="MLflow" src="https://img.shields.io/badge/Logging-MLflow-blue"></a>
<a href="https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.loggers.wandb.html#wandb"><img alt="Weights&Biases" src="https://img.shields.io/badge/Logging-Weigths%26Biases-yellow"></a>
<a href="https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.loggers.tensorboard.html"><img alt="Tensorboard" src="https://img.shields.io/badge/Logging-Tensorboard-FF6F00"></a>
</div>


This repository contains a framework for training deep learning-based classification and regression models 
with Pytorch Lightning. \
It comprises several architectures, regularization, augmentation and training techniques and
aims to provide easy-to-use baselines for experimenting with a lot of different setups. \
You can also integrate your own model and/or dataset and benefit from the features of this repository! \
Results of experiments on CIFAR-10 comparing different architectures in different training settings are shown below. \
Everything can be run via the Command Line Interface. You can choose one or multiple logger from Mlflow, Tensorboard and Weights&Biases for your experiment tracking. \
Training uses mixed precision and `torch.backends.cudnn.benchmark=True` by default to increase training speed. \
Best results are achieved with a PyramidNet using RandAugment augmentation, Shakedrop and Mixup.
It yields 0.986 Test Accuracy on CIFAR-10 and 0.875 Test Accuracy on CIFAR-100. \
Detailed results and used configurations can be seen in [CIFAR Results](#cifar-results).


# Table of Contents

* [How to run](#how-to-run)
  * [Requirements](#requirements)
  * [General instructions](#general-instructions)
  * [Available models and parameters](#available-models-and-parameters)
    * [Models](#models)
    * [Training Settings](#training-settings)
    * [Data Settings](#data-settings)
    * [Regularization Techniques](#regularization-techniques)
  * [Mlflow](#mlflow)
* [Including custom pytorch models](#including-custom-pytorch-models)
* [Including other datasets](#including-other-datasets)
* [Add custom augmentations](#add-custom-augmentations)
* [CIFAR Results](#cifar-results)


# How to run

## Requirements
First install the requirements in a [virtual environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) by:

```shell
pip install -r requirements.txt
```

You might need to adapt the cuda versions for torch and torchvision specified in the requirements. 
Find a torch installation guide for your system [here](https://pytorch.org/get-started/locally/). 

## General instructions

Everything in this repository can be run by executing the ```main.py``` script with corresponding arguments.
In order to train a model, one needs to specify the path to the directory that contains the datasets (```data_dir```) and the
path to the directory where logs should be saved (```exp_dir```). You can e.g. train on CIFAR10/CIFAR100 or Imagenet.
If the specified dataset is not found in the specified ```data_dir```, it will be automatically downloaded (178 MB for CIFAR). Imagenet cannot be downloaded automatically, please see [Imagenet Download](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) for access.

See [Including custom datasets](#including-other-datasets) for instructions about using your own data. This repository also handles regression problems. For that simply add the ```--regression``` flag, it will use the MSE Loss instead of Cross Entropy.
Here is an example of the command line for training a ResNet34 on CIFAR10:

```shell
python main.py ResNet34 --data CIFAR10 --data_dir "Path/to/data" --exp_dir "Path/to/logs"
```

By default, no checkpoints are saved! If you want to save the trained model you can use the ```--save_model``` flag. 
The model checkpoint will then be saved in the ```exp_dir``` along with the logs. You can adapt the name of the 
file by specifying ```--chpt_name "your-file-name"```. By default, the model is trained on one GPU. You can adapt 
this by setting e.g. ```--gpu_count 2``` to train on multiple GPUs using ddp strategy. If you want to use the CPU instead, set ```--gpu_count 0```.
Use the ```--suppress_progress_bar``` flag for not showing the progress bar during training. By default, only model accuracy is tracked. You can specify other metrics by adding them in the command line. Set ```--metrics acc f1 pr top5acc``` for tracking Accuracy, F1-Score, Precision & Recall and the Top-5 Accuracy. For regression ```--metrics mse mae``` are available.

## Available models and parameters

### Models
#

The following models are available:
* [PyramidNet](https://arxiv.org/pdf/1610.02915.pdf)
  * PyramidNet272 (depth 272, alpha 200)
  * PyramidNet110 (depth 110, alpha 270)
* [WideResNet](https://arxiv.org/pdf/1605.07146.pdf)
  * WRN2810 (depth 28, widen factor 10)
* [ResNet](https://arxiv.org/pdf/1512.03385.pdf)
  * ResNet152
  * ResNet50
  * ResNet34
  * ResNet18
* [PreActResNet](https://arxiv.org/pdf/1603.05027.pdf)
  * PreActResNet152
  * PreActResNet101
  * PreActResNet50
  * PreActResNet34
  * PreActResNet18
* [VGG](https://arxiv.org/pdf/1409.1556.pdf)
  * VGG16 (uses batch norm, does not include the fully connected layers at the end)
  * VGG19 (uses batch norm, does not include the fully connected layers at the end)
  
If you want to include your own model please see [Including custom models](#including-custom-pytorch-models) for instructions.

#
### Training Settings
#

By default, the following training settings are used:

* Epochs: 200 | ```--epochs 200```
* Batch Size: 128 | ```--batch_size 128```
  * If you set the number of GPUs > 1, your effective batch size becomes gpu_count * batch_size
* Optimizer: SGD (momentum=0.9, nesterov=False) | ```--optimizer SGD```
  * for enabling nesterov use the ```--nesterov``` flag (SGD only)
  * other available optimizers are:
    * [Madgrad](https://arxiv.org/pdf/2101.11075v2.pdf) | ```--optimizer Madgrad```
    * [Adam](https://arxiv.org/pdf/1412.6980.pdf) | ```--optimizer Adam```
    * [AdamW](https://arxiv.org/pdf/1711.05101v3.pdf) | ```--optimizer AdamW```
    * [RmsProp](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) | ```--optimizer Rmsprop```
    
* Learning Rate: 0.1 | ```--lr 0.1```
* Weight Decay: 5e-4 | ```--wd 5e-4```
* Scheduler: None
  * available LR schedulers are:
    * [Cosine Annealing](https://arxiv.org/pdf/1608.03983v5.pdf) | ```--scheduler CosineAnneal```
      * includes [Gradual Warmup](https://arxiv.org/pdf/1706.02677.pdf) | ```--warmstart 10```
        * linearly increases the LR to the initial LR for specified amount of epochs
        * Parameter: Int - Number of epochs to warm up
        * Default: 0
        * Value used in experiments: 10
    * MultiStep (multiply LR with 0.1 after 1/2 and 3/4 of epochs) | ```--scheduler MultiStep```
    * Step (multiply LR with 0.1 every 1/4 of epochs) | ```--scheduler Step```

* Seed: None
  * Specify a seed with ```--seed your_seed```
  * Disables cudnn.benchmark flag (required to make training deterministic) which results in slower training
* Number of workers in dataloaders: 12 | ```--num_workers 12```
* Number of GPUs | ```--gpu_count 1```
  * If > 1 training will be executed on multiple GPUs following ddp strategy
  * If 0 training will be executed on CPU

<!--
Run ```
python main.py -h``` for a description of all possible parameters.
-->
#
### Data Settings
#

Additionally, you can adapt the following parameters to your data.
* Dataset | ```--data CIFAR10```
  * You can choose between CIFAR10, CIFAR100 and Imagenet
  * See [Including custom datasets](#including-other-datasets) for using your own data
  * See [here](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) for instructions to download Imagenet
* Number of classes | ```--num_classes 10```
  * If you train on CIFAR or Imagenet this parameter will default to the correct number of classes, otherwise you have to specify it
* Set the task to Regression | ```--regression```
  * will use the MSE Loss instead of Cross Entropy
  * sets num_classes to 1
* Augmentation | ```--augmentation baseline```
  * Different augmentations depending on dataset
    * CIFAR:
      * Baseline (Random Crop, Random Horizontal Flips and Normalization) | ```--augmentation baseline```
      * Baseline + [Cutout](https://arxiv.org/pdf/1708.04552.pdf) (cutout size 16 for CIFAR10 and size 8 for CIFAR100) | ```--augmentation baseline_cutout```
      * Baseline + Cutout + [AutoAugment](https://arxiv.org/pdf/1805.09501.pdf) | ```--augmentation autoaugment```
      * Baseline + Cutout + [RandAugment](https://arxiv.org/pdf/1909.13719.pdf) | ```--augmentation randaugment```
    * Imagenet:
      * Baseline (Random Resized Crop (224), Random Horizontal Flip and Normalization) | ```--augmentation baseline```
      * Baseline + [Cutout](https://arxiv.org/pdf/1708.04552.pdf) (cutout size 112) | ```--augmentation baseline_cutout```
      * Baseline + [AutoAugment](https://arxiv.org/pdf/1805.09501.pdf) | ```--augmentation autoaugment```
      * Baseline + [RandAugment](https://arxiv.org/pdf/1909.13719.pdf) | ```--augmentation randaugment```
* Number of Input Dimensions | ```--input_dim 2```
  * At the moment only available for ResNets and VGGs
  * Specifies the number of dimensions of your data and chooses dynamically the corresponding 1D, 2D or 3D model operations
* Number of Input Channels | ```--input_channels 3```
  * At the moment only available for ResNets and VGGs
  * Specifies the number of channels of your data, e.g. 1 for grayscale images or 3 for RGB, and adapts the model architecture accordingly
* Very small images | ```--cifar_size```
  * At the moment only available for ResNets and VGGs
  * If used, more lightweight architectures designed for smaller images like CIFAR are used
  * if data is CIFAR this flag is activated by default


#
### Regularization Techniques
#

Moreover, there are several additional techniques available that are all disabled by default. 
<!--
In order to enable e.g. a WideResNet 28-10
with mixup, stochastic depth, label smoothing and a Cosine Annealing scheduler with a warmstart of 10 epochs you can run:

```shell
python main.py WRN2810 --scheduler CosineAnneal --warmstart 10 --mixup --stochastic_depth 0.1 --label_smoothing 0.1
```
-->
The following techniques are available for all models:
* [Label Smoothing](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf) | ```--label_smoothing 0.1```
    * Reduces overfitting and overconfidence by mixing the labels with the uniform distribution
    * Parameter: Float - determines amount of smoothing
    * Range: 0.0 (no smoothing) - 1.0 (full smoothing)
    * Default: 0.0
    * Value used in experiments: 0.1
* [Mixup](https://arxiv.org/pdf/1710.09412.pdf) | ```--mixup --mixup_alpha 0.2```
    * Data augmentation technique that generates a weighted combination of random image pairs from the training data
    * Parameter: mixup: bool (apply mixup or not), mixup_alpha: Float - controls the strength of interpolation between feature-target pairs
    * Range: 0.0 (no mixup) - 1.0 (very heavy mixup)
    * Default: 0.2 if ```--mixup``` is specified, otherwise 0.0
    * Value used in experiments: 0.2 (if mixup was used)
* [Sharpness Aware Minimization](https://arxiv.org/pdf/2010.01412.pdf) | ```--SAM```
    * Additional optimization that uses any optimizer as base optimizer but specifically seeks parameters that lie in neighborhoods having uniformly low loss
    * Parameter: bool
    * SAM trains significantly longer as it requires two forward-backward passes for each batch and disables 16-bit precision
    * It is recommended to train half of the epochs you would usually do
* [Disable weight decay for batchnorm parameters as well as bias](https://arxiv.org/pdf/1812.01187.pdf) | ```--undecay_norm```
    * Applies weight decay only to weights in convolutional and fully connected layers
    * Parameter: bool


The following techniques can only be used with ResNet-like models (including PyramidNets and WideResNets):

* [Stochastic Depth](https://arxiv.org/pdf/1603.09382.pdf) | ```--stochastic_depth 0.1```
    * Randomly drops some building blocks in ResNet-like models
    * Parameter: Float - determines the drop probability
    * Range: 0.0 (no dropping) - 1.0 (drop all)
    * Value used in experiments: 0.1
* [Shakedrop](https://arxiv.org/pdf/1802.02375.pdf) | ```--shakedrop```
    * Shake: Applies random perturbation by noise addition to the output of a residual branch
    * Drop: Applies Shake randomly starting with very low probability and linearly increasing probability up to 0.5 in final layer
    * Parameter: bool
    * Authors recommend training 6 times longer with shakedrop than you would normally do
* [Squeeze and Excitation](https://arxiv.org/pdf/1709.01507.pdf) | ```--se```
    * Adds a Squeeze & Excitation Layer to the end of the encoding part of the network
    * Squeeze: Aggregates feature maps across their spatial dimensions to an embedding allowing all layers to use the global receptive field of the network
    * Excitation: Takes the embedding as input and produces per-channel modulation weights, so that it enables the network to adapt the weighting of CNN feature maps
    * Parameter: bool
* [Final Layer Dropout](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) | ```--final_layer_dropout 0.5```
    * Applies Dropout right before the network's final output
    * Parameter: Float - determines the drop probability
    * Range: 0.0 (no dropout) - 1.0 (always drop)
    * Value used in experiments: 0.5
* [Zero initialization of the residual](https://arxiv.org/pdf/1812.01187.pdf) | ```--zero_init_residual```
    * Initializes the last layer (most often batchnorm) of a residual block with zeros so that residual blocks return their inputs (easier to train in the initial state)
    * Parameter: bool
* [Bottleneck](https://arxiv.org/pdf/1512.03385.pdf) | ```--bottleneck```
    * Using bottleneck building blocks instead of basic building blocks
    * Parameter: bool
    * Although bottleneck blocks are standard for deeper ResNets we have empirically found that deeper ResNets with basic blocks outperform their respective bottleneck versions on Cifar

#
## Logging

You can choose one or multiple logger. [MLFlow](https://mlflow.org/), [Tensorboard](https://www.tensorflow.org/tensorboard) and [Weights&Biases](https://wandb.ai/site) are available. Use the ```--logger``` flag to specify your logger, e.g. 
```
--logger mlflow tensorboard wandb
``` 
will use all, while 
```
--logger wandb
```
will only use Weights&Biases. By default, only MLFlow is used.

In the logging interfaces you can see all your runs and corresponding metrics. You can analyse your runs there or download them as a csv file for further analysis.

Moreover, a confusion matrix is logged, by default, for your validation set. You can additionally log a confusion matrix for training by
```
--confmat all
```
or disable confusion matrix logging with
```
--confmat disable
```


For viewing your logs navigate to the log directory (specified with the ```--exp_dir``` flag) and go into the directory named after your dataset. Depending on what logger you chose, there are different ways to view the logs:

### MLflow 

```shell
mlflow ui
```

### Tensorboard

```shell
tensorboard --logdir tensorboard
```

### Weights&Biases

A link will be displayed at the beginning of the training that will lead to the wandb interface. If you have an account then logs will be automatically synced. 


# Including custom pytorch models


In order to make the functionalities of this repository available for your model you can use the ```ModelConstructor```.\
Follow these steps:
1. Choose a name for your model, e.g. ```CustomModel```, that you want to use in the Command Line Interface and that will be logged.
2. Register your model name by adding it to the list ```registered_models``` in ```utils.py``` and import your model at the top of the file.
3. In the same file adapt the ```get_model``` function as follows:
```python
def get_model(model_name, params, num_classes):
    
    ...
    
    # use the model name you chose
    elif model_name == 'CustomModel':
        
        # initialize your model
        # pass any additonal parameters you need for your model here, e.g. the number of classes to predict
        custom_model = CustomModel(num_classes) 
    
        # pass your initialized model to the ModelConstructor
        model = ModelConstructor(custom_model, params)
        
    return model
```
That's it!
Note that the techniques Stochastic Depth, Shakedrop, Squeeze & Excitation, Final Layer Dropout and Bottleneck require architectural changes and cannot be 
made available automatically for a custom model. All other functionalities of this repository are now also available for your model.
You can now train your model e.g. with RandAugment, Mixup, Madgrad Optimizer, Sharpness Aware Minimization and a Cosine Annealing LR scheduler with warmstart of 5 epochs all out of the box like this:
```shell
python main.py CustomModel --lr 0.0001 --optimizer Madgrad --scheduler CosineAnneal --warmstart 5 --mixup --augmentation randaugment --SAM
```

# Including other datasets

If you want to train on your own data, you can easily integrate it into this repository. \
Follow these steps:
1. In the ```dataset``` directory create a new file that implements the torch dataset class for your data.
2. Choose a name for your dataset and add it to the ```registered_datasets``` in ```utils.py```.
3. In the ```augmentation/policies``` directory there are files with specific augmentation policies for each dataset. You can add one with custom augmentations for your dataset if you want. If you want to use policies that are used on CIFAR or Imagenet you do not need to add a file here.
4. Adapt ```base_model.py```\
4.1 Import your data class from step 1.\
4.2 In the ```__init__``` function of the ```BaseModel``` there is a dataset specific part that chooses the augmentations. Add a new ```elif``` clause that checks for your custom dataset name and assign your augmentations:
    ```python
    class BaseModel(pl.LightningModule):
        def __init__(self):
          
            ...
            
            # use your dataset name
            elif self.dataset == 'CustomData':
                self.mean, self.std = <mean_and_standard_deviations>
                from augmentations.policies.custom_data import custom_aug1, custom_aug2, custom_test_transform

                # transformations that will be used for training
                if self.aug == 'aug1':
                    # assumes that custom_aug1 is a function that takes mean and std and returns a composed augmentation pipeline
                    self.transform_train = custom_aug1(self.mean, self.std)
                
                elif self.aug == 'aug2':
                    self.transform_train = custom_aug2(self.mean, self.std)

                # transformations that will be used for validation and test data
                self.test_transform = custom_test_transform(self.mean, self.std)
              
      ```
    4.3 In the ```train_dataloader``` and ```val_dataloader``` functions of the ```BaseModel``` again add a new ```elif``` clause that checks for the dataset name and initialize your dataset with the transforms that you assigned ealier:
    ```python
    class BaseModel(pl.LightningModule):
        def train_dataloader(self):
          
            ...
            
            # use your dataset name
            elif self.dataset == 'CustomData':
                trainset = CustomDataset(..., transform=self.transform_train)
              
      ```
      Do the same for the ```val_dataloader```.\
      That's it! You can now use all training pipelines, regularization techniques and models with your dataset.

# Add custom augmentations
See [Including other datasets](#including-other-datasets) if you want to use a new dataset including specific augmentations. If you only want to try out new augmentations for already included datasets like CIFAR or Imagenet you can simply add them to the respective dataset file. For CIFAR this would be augmentation/policies/cifar.py. See 4.2 of [Including other datasets](#including-other-datasets) on how to integrate the new augmentation policy.

# CIFAR Results

The following experiments aim to show general tendencies for different training settings and techniques. In order 
to be consistent with the literature we evaluate all our experiments on the respective test sets. We are aware of 
problems resulting from this experimental setup and would like to emphasize that test set optimization should never 
be done when developing new methods or making claims about the generality of the results. A proper experimental setup 
should always include a validation set which is used for hyperparameter optimization. Thus, all of the following 
experiments are merely intended to provide general tendencies without any guarantee that they will generalize to unseen 
data.

### Best test accuracies by each model

![](imgs/cifar10_best_runs.png?raw=true "Best runs for each model on CIFAR-10")

The following table shows the exact configurations that were used to achieve the stated test accuracies on CIFAR-10 and CIFAR-100.
Note that not all possible combinations of parameters were tested. Therefore, one shouldn't draw any conclusions about single
parameters. The table only aims to enable reproducibility. 
Please see the experiments in the following sections for an evaluation of individual techniques. 



| Model        | epochs|LR  | Optimizer  |Nesterov|Scheduler| Warmstart|Weight Decay|Augmentation|Undecay Norm|Label Smoothing|Stochastic Depth|Mixup|Final Layer Dropout|Squeeze & Excitation|Shakedrop|Zero-Init-Residual|CIFAR10 Test Acc|CIFAR100 Test Acc|
| -------------|:---:|:-----:| :-----:| :-----:| :-----:| :-----:| :-----:| :-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| PyramidNet110 | 1800| 0.05   | SGD|False|CosineAnneal  | 0 | 5e-5 | RandAugment |False| 0.0 | 0.0 | True| 0.0|False|True|False|0.9864|0.8746
| PyramidNet272 | 1800| 0.05   | SGD|False|CosineAnneal  | 0 | 5e-5 | AutoAugment |False| 0.0 | 0.0 | False| 0.0|False|True|  False|0.9836|0.8728
| WRN2810 | 1800| 0.1   | SGD|False|CosineAnneal  | 0 | 5e-4 | AutoAugment |False| 0.0 | 0.0 | False| 0.0|False|False|False|  0.9775|0.8407
| PreActResNet152 | 200| 0.1   | SGD - SAM|False|CosineAnneal  | 10 | 5e-4 | RandAugment |False| 0.0 | 0.0 | False| 0.0|False|False| False| 0.9741|0.8310
| ResNet152 | 200| 0.1   | SGD |True|CosineAnneal  | 0 | 5e-4 | AutoAugment |False| 0.0 | 0.0 | False| 0.0|False|False| True| 0.9739|0.8249
| PreActResNet50 | 200| 0.1   | SGD - SAM|False|CosineAnneal  | 10 | 5e-4 | RandAugment |False| 0.0 | 0.0 | False| 0.0|False|False| False| 0.9739|0.8247
| ResNet50 | 200| 0.1   | SGD - SAM|False|CosineAnneal  | 0 | 5e-4 | AutoAugment |False| 0.0 | 0.0 | False| 0.0|False|False| False| 0.9735|0.8294
| PreActResNet34 | 400| 0.1   | SGD |False|CosineAnneal  | 10 | 5e-4 | AutoAugment |False| 0.0 | 0.0 | False| 0.0|False|False| False| 0.9730|0.8129
| ResNet34 | 200| 0.1   | SGD - SAM|False|CosineAnneal  | 0 | 5e-4 | RandAugment |False| 0.0 | 0.0 | False| 0.0|False|False| False| 0.9704|0.8195
| PreActResNet18 | 400| 0.1   | SGD |False|CosineAnneal  | 10 | 5e-4 | RandAugment |False| 0.0 | 0.0 | False| 0.0|False|False| False| 0.9680|0.7897
| ResNet18 | 200| 0.1   | SGD - SAM|False|CosineAnneal  | 0 | 5e-4 | RandAugment |False| 0.0 | 0.0 | False| 0.0|False|False| False| 0.9663|0.8038
| VGG16 | 400| 0.1   | SGD |False|CosineAnneal  | 0 | 5e-4 | AutoAugment |False| 0.0 | 0.0 | False| 0.0|False|False| False| 0.9599|0.7782


### Optimizer, Learning Rate and Scheduler

The choice of an optimizer, corresponding learning rate and learning rate scheduler all have a large impact on 
convergence speed and model performance. Here different setups were compared for the ResNet34 model. All versions were 
trained for 200 epochs, using a batch size of 128, AutoAugment augmentation and a weight decay of 4e-5. 
No other regularization techniques or additional settings were used. 
SGD with momentum of 0.9 and no nesterov was compared to Adam (no amsgrad), AdamW (no amsgrad), RMSProp and the recently introduced Madgrad. 
In addition, Sharpness Aware Minimization (SAM) was used with SGD and Madgrad as base optimizers.
The considered learning rate schedulers were Step (multiply lr with 0.1 evey 50 epochs), MultiStep (multiply lr with 0.1 after 1/2 and 3/4
of epochs) and Cosine Annealing without warmstart. Every combination of LR, optimizer and scheduler that is not displayed yielded a score lower than
0.92 and is not shown to give a more clear view on the good performing setups. \
For reproducing e.g. the best SGD - SAM run which yielded more than 0.97 accuracy you can run:

```shell
python main.py ResNet34 --lr 0.1 --optimizer SGD --SAM --scheduler CosineAnneal --augmentation autoaugment --epochs 200
```

It can be seen that optimizer and LR are highly dependent on each other. While SGD performs better with higher LRs like 0.1, 
the other ones yield better results with much lower LRs. Madgrad yields best results with the smallest LR tested (0.0001).
Moreover, Sharpness Aware Minimization (SAM) always outperformed the respective base optimizer alone. 
Regardless of the optimizer the best scheduler was usually Cosine Annealing.

![](imgs/lr-opt-sched.png?raw=true "ResNet34 performance across LRs, schedulers and optimizers")


### Augmentation techniques and Sharpness Aware Minimization (SAM)

Baseline augmentation on CIFAR involves Random Cropping, Random Horizontal Flips (with a 0.5 probability) and Normalization.
In addition, Cutout (with a size of 16 on CIFAR-10 and 8 on CIFAR-100) is added in baseline_cutout. 
It erases a random patch of the image and is able to improve the robustness of the model by simulating occlusions. 
AutoAugment and RandAugment both include the baseline augmentation, then their respective policy and finally also utilize cutout.
While AutoAugment is a set of learned augmentation policies that work well on CIFAR, RandAugment does not have to be learned first 
but applies a subset of transformations randomly. 

Here all models were using either SGD or SGD with SAM as an optimizer, an initial LR of 0.1 and a Cosine Annealing scheduler.
While the SGD models were trained for 400 epochs, the SAM models were trained for 200 epochs to ensure a fair comparison 
(in terms of total training time).
The PreActResNets were all using a gradually increasing warmstart for 10 epochs before starting to decay the LR, because otherwise 
the LR of 0.1 would have been too high for them in the beginning, so they would result in NaNs. The other models did not use a warmstart.
If you e.g. want to reproduce the results of a PreActResNet50 with RandAugment and SGD - SAM optimizer, you can run the following:

```shell
python main.py PreActResNet50 --lr 0.1 --optimizer SGD --SAM --scheduler CosineAnneal --warmstart 10 --augmentation randaugment --epochs 200
```
The respective run with SGD alone would be:
```shell
python main.py PreActResNet50 --lr 0.1 --optimizer SGD --scheduler CosineAnneal --warmstart 10 --augmentation randaugment --epochs 400
```
Single runs are shown for each augmentation technique with their resulting test accuracy. Adding Cutout to the baseline 
considerably improves the performance of all models. AutoAugment and RandAugment yield very similar results but both consistently
outperform the baselines with and without cutout. 
Moreover, when comparing the performance of the optimizers it can be seen that the influence of SAM varies between augmentations
and model depth. In the case of baseline and cutout augmentation using SAM always improved the final test accuracy significantly, regardless of the model.
However, when using heavier augmentation techniques like AuoAugment and RandAugment SAM only improved performance of deeper models.
VGG16 as well as the (PreAct)ResNets18 and 34 all did not improve using SAM or even performed worse.
On the other hand the PyramidNets and deeper ResNets still yielded better scores when utilizing SAM. 



![](imgs/aug.png?raw=true "Augmentation techniques across models and optimizers")

### Training time comparison




Here the previous runs using AutoAugment are shown with their respective time needed to complete a full training.
In addition, SGD runs with only 200 epochs are depicted. 
All models were trained on a GeForce RTX 2080 Ti.
As expected smaller models usually need less training time but yield lower accuracy. 
The WideResNet 28-10 however, trains faster than the ResNet152 and PreActResNet152 but still performs better.
Training the plain SGD versions for 400 instead of 200 epochs always increases the performance significantly 
except for the ResNet152 and PreActResNet152. As seen in the previous section, SAM used with more complex models 
outperforms the 400 epoch SGD runs. Therefore, when using SAM training for only half of the epochs you would usually do
can be a good trade-off between performance and training time. 







![](imgs/time.png?raw=true "Model Training Time Comparison")

### Adding other techniques

There are several architectural tweaks such as e.g. stochastic depth, general regularization techniques (e.g. mixup) 
and other tricks like using a LR scheduler warmstart or using nesterov accelerated SGD. Find a short description, links to papers
and an explanation how they were used for all techniques in [Available models and parameters](#available-models-and-parameters).
In order to test their effects on the CIFAR datasets each of them was added individually to a baseline model. 
Since final test accuracies can vary for similar runs, all models were trained 5 times. 
The baseline models were trained for 200 epochs, using SGD with a LR of 0.1, a Cosine Annealing scheduler and AutoAugment augmentation. 
The mean test accuracies are depicted by their respective dashed vertical lines. The results of the runs including each 
respective technique are represented by boxplots to show the variance within runs. An example run for a WideResNet using mixup
would be:
```shell
python main.py WRN2810 --lr 0.1 --optimizer SGD --scheduler CosineAnneal --augmentation autoaugment --epochs 200 --mixup
```
It can be seen that variance in runs is higher for some techniques such as squeeze & excitation while it is rather stable 
for other techniques such as nesterov or mixup. Moreover, different models work differently well with some techniques like
final layer dropout which mostly increases performance of the WideResNet while it harms performance of the other two models.
For all three models not decaying batchnorm and bias decreased test accuracy while using a warmstart, mixup and nesterov 
all increased performance. 

![](imgs/techniques.png?raw=true "Other techniques across models")

# Acknowledgements

<p align="left">
  <img src="imgs/Logos/HI_Logo.png" width="150"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="imgs/Logos/DKFZ_Logo.png" width="500"> 
</p>

This Repository is developed and maintained by the Applied Computer Vision Lab (ACVL)
of [Helmholtz Imaging](https://www.helmholtz-imaging.de/).