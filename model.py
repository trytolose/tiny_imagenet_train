import torch
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet


class Resnet_v1(nn.Module):
    def __init__(self, base_resnet_model):
        super().__init__()
        
#         model = models.resnet18()
        head = nn.Sequential(nn.Conv2d(3, 64, 7, stride=1, padding=3, bias=False), 
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        self.newmodel = nn.Sequential(head, *(list(base_resnet_model.children())[4:-1]))
        self.fc = nn.Linear(512,200)    

    def forward(self, x):
        
        x = self.newmodel(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Resnet_v2(nn.Module):
    def __init__(self, base_resnet_model, dropout_p=0.5):
        super().__init__()
        
#         model = models.resnet18()
        head = nn.Sequential(nn.Conv2d(3, 64, 7, stride=1, padding=3, bias=False), 
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        self.newmodel = nn.Sequential(head, *(list(base_resnet_model.children())[4:-1]))
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(512,200)   

    def forward(self, x):
        
        x = self.newmodel(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x



    
def get_model(args):

    if 'effnet-b0' in args.model_name:
        backbone = EfficientNet.from_name('efficientnet-b0')
        backbone._fc = nn.Linear(1280, 200)
        return backbone

     
    if 'resnet18' in args.model_name:
        backbone = models.resnet18()
        backbone.fc = nn.Linear(512,200)
        
    if 'resnet34' in args.model_name:
        backbone = models.resnet34()
        backbone.fc = nn.Linear(512,200)
    
    if 'wo_first_pool' in args.model_name:
        model = Resnet_v1(backbone)
        return model
        
    if 'wo_first_pool_dropout' in args.model_name:
        model = Resnet_v2(backbone, dropout_p=0.5)
        return model

    
    return backbone