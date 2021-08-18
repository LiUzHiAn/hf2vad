import torch
import torch.nn as nn

def weights_init_kaiming(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight.data, a=0.1, mode='fan_in')
    elif isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight.data, a=0.1, mode='fan_in')
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
