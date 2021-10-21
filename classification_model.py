"""
pytorch models.
installation for CUDA 10.0:
1. pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-win_amd64.whl
2. pip install pip install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp37-cp37m-win_amd64.whl
3. pip install segmentation-models-pytorch
4. pip install torchsummary
"""
import os
import torch
from typing import Optional, Union, List
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from segmentation_models_pytorch.encoders.densenet import densenet_encoders
from torch.autograd import Variable
import numpy as np
encoders = {}
encoders.update(densenet_encoders)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class ClassificationHead(nn.Sequential):

    def __init__(self, num_ftrs, out_channels):
        super().__init__()
        self.linear = nn.Linear(num_ftrs, out_channels)
        self.num_ftrs = num_ftrs
        self.sigmoid = nn.Sigmoid()

        self.classifier = nn.Sequential(
            # nn.Linear(num_ftrs, 1024),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            # nn.Linear(4096, 4096),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(num_ftrs, 128), # out_channels
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, out_channels)
        )

    def forward(self, *features):
        x = features[-1]
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (6, 6))
        x = torch.flatten(x, 1)
        # x = x.view(-1, self.num_ftrs)
        x = self.classifier(x)

        return x

class ClassificationModel(torch.nn.Module):
    def initialize(self):
        self.initialize_head(self.classification_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)

        labels = self.classification_head(*features)

        return labels

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`
        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)
        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)
        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x

    def initialize_head(self, module):
        for m in module.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



class classification_model(ClassificationModel):
    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_depth: int = 5,
                 encoder_weights: str = "imagenet",
                 decoder_channels: List[int] = (256, 128, 64, 32, 16),
                 in_channels: int = 3,
                 classes: int = 1,
                 activation: str = 'softmax'):
        super(classification_model, self).__init__()

        # encoder
        self.encoder = self.get_encoder(encoder_name, in_channels=in_channels, depth=encoder_depth,
                                        weights=encoder_weights)
        #ToDo: change num_features to the exact num
        self.classification_head = ClassificationHead(num_ftrs=1024*6*6,
                                                      out_channels=classes)
        self.middle_layer = []
        self.name = 'u-{}'.format(encoder_name)
        # self.initialize_classification_model()

    def forward(self, x):
        features = self.encoder(x)
        output = self.classification_head(*features)
        return output
    def parameters_to_grad(self):
        return self.parameters()

    def get_encoder(self, name, in_channels=3, depth=5, weights=None):
        Encoder = encoders[name]["encoder"]
        params = encoders[name]["params"]
        params.update(depth=depth)
        encoder = Encoder(**params)

        if weights is not None:
            if not os.path.exists(weights):
                settings = encoders[name]["pretrained_settings"][weights]
                encoder.load_state_dict(model_zoo.load_url(settings["url"]))
            else:
                state_dict = torch.load(weights, map_location='cpu')
                state_dict["classifier.bias"] = []
                state_dict["classifier.weight"] = []
                encoder.load_state_dict(state_dict)

        encoder.set_in_channels(in_channels)

        return encoder

class CombinedModel(ClassificationModel):
    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_depth: int = 5,
                 encoder_weights: str = "imagenet",
                 decoder_channels: List[int] = (256, 128, 64, 32, 16),
                 in_channels: int = 3,
                 classes: int = 1,
                 activation: str = 'softmax',device='cpu'):

        super(CombinedModel, self).__init__()

        # encoder
        self.encoder_base = self.get_encoder(encoder_name, in_channels=in_channels, depth=encoder_depth,
                                        weights=encoder_weights)
        self.encoder = self.get_encoder(encoder_name, in_channels=in_channels, depth=encoder_depth,
                                             weights=encoder_weights)
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.encoder_depth = encoder_depth
        self.device = device
        #ToDo: change num_features to the exact num
        self.classification_head = ClassificationHead(num_ftrs=1024*6*6,
                                                      out_channels=classes)
        self.blocks = [[],[],[],[],[]]
        for (name,p1),(name2,p2) in zip(self.encoder_base.named_parameters(),self.encoder.named_parameters()):
            assert name == name2
            if name in ['features.conv0.weight','features.norm0.weight','features.norm0.bias' ]:
                self.blocks[0].append((name,p1,p2))
            elif 'denseblock1' in name or 'transition1' in name:
                self.blocks[1].append((name,p1,p2))
            elif 'denseblock2' in name or 'transition2' in name:
                self.blocks[2].append((name,p1,p2))
            elif 'denseblock3' in name or 'transition3' in name:
                self.blocks[3].append((name,p1,p2))
            elif 'denseblock4' in name or name in ['features.norm5.weight','features.norm5.bias' ]:
                self.blocks[4].append((name,p1,p2))
            else:
                print(name)
                raise Exception()
        self.middle_layer = []
        for i in range(len(self.blocks)):
            w = torch.nn.Parameter(torch.tensor(np.random.normal(loc=(i-5)/2)))
            self.register_parameter(name=f'w{i}', param=w)
            self.middle_layer.append(w)


    def forward(self, x):
        empty_encoder = self.get_encoder(self.encoder_name, in_channels=self.in_channels, depth=self.encoder_depth)
        empty_encoder.to(self.device)
        new_state_dict = empty_encoder.state_dict()
        for block, middle in zip(self.blocks,self.middle_layer):
            for layer in block:
                assert layer[0] in new_state_dict
                w1 = torch.sigmoid(middle)
                new_state_dict[layer[0]] = layer[1] * (1-w1) + layer[2] * w1
        for name,p in empty_encoder.named_parameters():
            p.requires_grad = False
            p.copy_(new_state_dict[name])
        features = empty_encoder(x)
        output = self.classification_head(*features)
        return output
    def parameters_to_grad(self):
        return [{'params':list((self.encoder.parameters())),'lr':0.001},{'params':self.middle_layer,'lr':0.1}]
    def get_encoder(self, name, in_channels=3, depth=5, weights=None):
        Encoder = encoders[name]["encoder"]
        params = encoders[name]["params"]
        params.update(depth=depth)
        encoder = Encoder(**params)

        if weights is not None:
            if not os.path.exists(weights):
                settings = encoders[name]["pretrained_settings"][weights]
                encoder.load_state_dict(model_zoo.load_url(settings["url"]))
            else:
                state_dict = torch.load(weights, map_location='cpu')
                state_dict["classifier.bias"] = []
                state_dict["classifier.weight"] = []
                encoder.load_state_dict(state_dict)

        encoder.set_in_channels(in_channels)

        return encoder



