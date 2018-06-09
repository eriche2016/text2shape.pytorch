import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from IPython.core.debugger import Tracer 
deug_here = Tracer() 

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

class  Conv3d_Pdmd(nn.Module): 
    """
    wrapping nn.conv3d and implement 'same' padding style 
    """
    def __init__(self, D_in, D_out, ks, st, pd, padding_mode=None, bias=True): 
        super(Conv3d_Pdmd, self).__init__()
        # if padding_mode is specified to be 'same' or 'valid', or 'full'
        # then padding size is computed by ks, here we donot specify pd 
        if padding_mode: 
            pd = pad_size(ks, padding_mode) 

        self.conv = nn.Conv3d(D_in, D_out, ks, st, pd, bias=bias) 

        # initialize 
        init.kamming_normal(self.conv.weight)
        if bias: 
            init.constant(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x) 


# define network1
class Classifier1(nn.Module):
    def __init__(self, num_classes): 
        super(Classifier1, self).__init__() 
        
        self.feature = nn.Sequential(
            Conv3d_Pdmd(1, 64, (3, 3, 3), strides=(2, 2, 2), pading_mode='same'),  
            nn.BatchNorm3d(64), 
            nn.ReLU(True),

            Conv3d_Pdmd(64, 128, (3, 3, 3), strides=(2, 2, 2), pading_mode='same'),  
            nn.BatchNorm3d(128), 
            nn.ReLU(True), 
 
            Conv3d_Pdmd(128, 256, (3, 3, 3), strides=(2, 2, 2), padding_mode='same'),  
            nn.BatchNorm3d(256), 
            nn.ReLU(True), 
            )
        self.fc1 = nn.Linear(256, num_classes)

    def forward(self, x):
        conv_feats = self.feature(x) # bz x num_features x H x L x W
        # average pooling 
        out = F.avg_pool3d(conv_feats, conv_feats.size()[2:])
        encoder_output = out 

        out = out.view(out.size(0), -1)
        out = self.fc1(out) 
        logits = out
        
        prob = F.softmax(out, dim=1)

        output_dict = {
        'logits': out,
        'probabilities': prob, 
        'encoder_output': encoder_output
        }
        
        return output_dict  


# define network1
class Classifier128(nn.Module):
    def __init__(self, num_classes): 
        super(Classifier128, self).__init__() 
        
        self.feature = nn.Sequential(
            Conv3d_Pdmd(1, 64, (3, 3, 3), strides=(2, 2, 2), pading_mode='same'),  
            nn.BatchNorm3d(64), 
            nn.ReLU(True),

            Conv3d_Pdmd(64, 128, (3, 3, 3), strides=(2, 2, 2), pading_mode='same'),  
            nn.BatchNorm3d(128), 
            nn.ReLU(True), 
 
            Conv3d_Pdmd(128, 256, (3, 3, 3), strides=(2, 2, 2), padding_mode='same'),  
            nn.BatchNorm3d(256), 
            nn.ReLU(True), 
            )
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        conv_feats = self.feature(x) # bz x num_features x H x L x W
        # average pooling 
        out = F.avg_pool3d(conv_feats, conv_feats.size()[2:])

        out = out.view(out.size(0), -1)
        out = self.fc1(out) 
        encoder_output = out # embeding the 3d shape into 128-dim vector
        out = self.fc2(out)
        logits = out
        
        prob = F.softmax(out, dim=1)

        output_dict = {
        'logits': logits,
        'probabilities': prob, 
        'encoder_output': encoder_output
        }
        return output_dict  

if __name__ == '__main__': 
    pass 


