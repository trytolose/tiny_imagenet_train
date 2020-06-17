I tried several NN architectures from scratch:  
1. Vanil Reset18 from torchvision  
2. Vanil Resnet34 from torchvision
3. Resnet18 without first MaxPool and stride=1 for first Conv layer with kernel=7
4. Resnet18 like previous with Dropout layer before last fully-connected layer  
5. Efficientnet-b0

Then I experimented with different optimizers(SGD, Adam, RAdam), tried smooth Losss, added default augmentations, also used cutmix and mixups. Also tried Autoaugmentation's Imagenet Policy, but it didn't work well. Here are the main results:

```
resnet18                       0.4079
resnet34                       0.4077
resnet18_v1                    0.5490
resnet18_v2                    0.5474
efficientnet_b0                0.4029

resnet18_v2_augs               0.5461
resnet18_v2 wd                 0.5986
resnet18_v2 wd cutmix          0.6343
resnet34_v2 wd cutmix          0.6401
```


Best configuration was:

Resnet34 without first pooling layer and stride=1 for first layer
Cross entropy loss
Adam optimizer with start_lr=0.001
weights_decay=0.0001
reduce on pleato scheduler 
150 epochs
Main augmentations: horizontal flip,  randomCrop(56x56), rotate, randomContrast, randomGamma
Cutmix augmentation inside minibatch 


For reproduce experiment process just run
```shell
bash experiments.sh
```


I hope it will work, although I changed training code during experiments.  

For reproducing best case run:
```shell
bash best.sh
```

All checkpoints and tensorboard logs are here: https://drive.google.com/file/d/1al4sYv0JOT45MSlWrAA6sBslXVW6YI3X/view?usp=sharing