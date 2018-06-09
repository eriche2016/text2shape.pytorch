import torch 
import torch.nn as nn 
from torch.autograd import Variable 
import numpy as np 

import pdb 

#################################################################
# original code, the author implement the model using tensorflow
# and the padding style is 'same'
# thus re-implement such style of padding in pytorch 
#################################################################
def pad_size(ks, mode):

  assert mode in ["valid","same","full"]

  if mode == "valid":
    return (0,0,0)

  elif mode == "same":
    assert all([ x % 2 for x in ks ])
    return tuple( x // 2 for x in ks )

  elif mode == "full":
    return tuple( x - 1 for x in ks )

class  Conv3d_TransposePdmd(nn.Module): 
    """
    wrapping nn.conv3d and implement 'same' padding style 
    currently only suport same
    """
    def __init__(self, D_in, D_out, ks, st, pd, pading_mode='same', bias=True): 
        super(Conv3d_Pdmd, self).__init__()
        # if padding_mode is specified to be 'same' or 'valid', or 'full'
        # then padding size is computed by ks, here we donot specify pd 
        if padding_mode: 
            pd = pad_size(ks, pading_mode) 

        self.conv = nn.ConvTranspose3d(D_in, D_out, ks, st, pd, bias=bias) 

        # initialize 
        init.kamming_normal(self.conv.weight)
        if bias: 
            init.constant(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x) 

##############################################################
## Define the Generator 
##############################################################
class Text2ShapeGenerator1(nn.Module):
    def __init__(self):
        super(Text2ShapeGenerator1, self).__init__() 
        self.fc1 = nn.Sequential( 
            nn.Linear(256, 512*4*4*4), 
            nn.BatchNorm1d(512*4*4*4),
            nn.ReLU(True)
        )  

        self.up_layers = nn.Sequential(
            Conv3d_TransposePdmd(512, 512, (4,4,4), strides=(1,1,1)),
            nn.BatchNorm3d(512),
            nn.ReLU(True),

            Conv3d_TransposePdmd(512, 256, (4,4,4), strides=(2,2,2)),
            nn.BatchNorm3d(256),
            nn.ReLU(True),

            Conv3d_TransposePdmd(256, 128, (4,4,4), strides=(2,2,2)),
            nn.BatchNorm3d(128),
            nn.ReLU(True),

            Conv3d_TransposePdmd(128, 4, (4,4,4), strides=(2,2,2), padding_mode='same'),
        )

        self.last_activation = nn.Sigmoid()

    def forward(self, x): 
        """
        input: text_encoding_with_noise
        """ 
        x = self.fc1(x)
        x = x.view(-1, 512, 4, 4, 4)
        logits = self.up_layers(x) 
        sigmoid_output = self.last_activation(logits) 

        output_dict = {
        'logits': logits, 
        "sigmoid_output": sigmoid_output
        }

        return output_dict

##############################################################
## Define the Discriminator 
##############################################################
#################################################################
# original code, the author implement the model using tensorflow
# and the padding style is 'same'
# thus re-implement such style of padding in pytorch 
#################################################################
def pad_size(ks, mode):

  assert mode in ["valid","same","full"]

  if mode == "valid":
    return (0,0,0)

  elif mode == "same":
    assert all([ x % 2 for x in ks ])
    return tuple( x // 2 for x in ks )

  elif mode == "full":
    return tuple( x - 1 for x in ks )

class Conv3d_Pdmd(nn.Module): 
    """
    wrapping nn.conv3d and implement 'same' padding style 
    """
    def __init__(self, D_in, D_out, ks, st, pd, pading_mode=None, bias=True): 
        super(Conv3d_Pdmd, self).__init__()
        # if padding_mode is specified to be 'same' or 'valid', or 'full'
        # then padding size is computed by ks, here we donot specify pd 
        if padding_mode: 
            pd = pad_size(ks, pading_mode) 

        self.conv = nn.Conv3d(D_in, D_out, ks, st, pd, bias=bias) 

        # initialize 
        init.kamming_normal(self.conv.weight)
        if bias: 
            init.constant(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x) 

class Text2ShapeDiscriminator2(nn.Module):
    def __init__(self):
        super(Text2ShapeDiscriminator2, self).__init__() 
        # For Shape 
        self.feature = nn.Sequential(
            Conv3d_Pdmd(4, 64, (4, 4, 4), strides=(2, 2, 2), pading_mode='same'),  
            nn.LeakyReLU(True),

            Conv3d_Pdmd(64, 128, (4, 4, 4), strides=(2, 2, 2), pading_mode='same'),  
            nn.LeakyReLU(True), 

            Conv3d_Pdmd(128, 256, (4, 4, 4), strides=(2, 2, 2), padding_mode='same'),  
            nn.LeakyReLU(True), 

            Conv3d_Pdmd(256, 512, (4, 4, 4), strides=(2, 2, 2), padding_mode='same'),  
            nn.LeakyReLU(True),

            Conv3d_Pdmd(512, 256, (2, 2, 2), strides=(2, 2, 2), padding_mode='same'),  
            nn.LeakyReLU(True), 
        )

        # After self.feature, we will reshape features for shape, so that we will concatenate with text features 


        # Text embedding input
        embedding_fc_dim = 256
        
        # Add FC layer
        self.net_emb_text_no_noise = nn.Sequential( 
            nn.Linear(256, embedding_fc_dim), 
            nn.LeakyReLU(True),
            nn.Linear(256, embedding_fc_dim), 
            nn.LeakyReLU(True),

         ) 

        # then we concatenate text and shape features 
        self.mix_text_shapes_net = nn.Sequential(
            nn.Linear(512, 128), 
            nn.LeakyReLU(), 
            nn.Linear(128, 64), 
            nn.LeakyReLU(), 
            nn.Linear(64, 1)
        )

        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        """"
        x: {shapes, text_emebddings}
        """
        shape_features = self.feature(x[1]) 
        # we view shape features to be 1-d 
        shape_features = shape_features.view(-1, 256)

        text_features = self.net_emb_text_no_noise(x[2])

        mix_features = torch.cat([shape_features, text_features], dim=1)
        logits =  self.mix_text_shapes_net(mix_features)

        sigmoid_output = self.last_activation(logits)

        return {'sigmoid_output': sigmoid_output, 'logits': logits}

if __name__ == '__main__': 
    pass 