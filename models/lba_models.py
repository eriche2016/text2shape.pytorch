#######################################################
## for LBA1 model 
#######################################################

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
# from torch.autograd import Variable 

from IPython.core.debugger import Tracer 
debug_here = Tracer() 


def pad_size_1d(ks, mode):

  assert mode in ["valid","same","full"]

  if mode == "valid":
    return 0

  elif mode == "same":
    assert  ks % 2 
    return ks // 2 

  elif mode == "full":
    return ks - 1 

class  Conv1d_Pdmd(nn.Module): 
    """
    wrapping nn.conv3d and implement 'same' padding style 
    currently only suport same
    """
    def __init__(self, D_in, D_out, ks, st=1, pd=0, padding_mode='same', bias=True): 
        super(Conv1d_Pdmd, self).__init__()
        # if padding_mode is specified to be 'same' or 'valid', or 'full'
        # then padding size is computed by ks, here we donot specify pd 
        if padding_mode: 
            pd = pad_size_1d(ks, padding_mode) 

        self.conv = nn.Conv1d(D_in, D_out, ks, st, pd, bias=bias) 

        # initialize 
        # init.kamming_normal(self.conv.weight)
        # if bias: 
        #    init.constant(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x) 


class Text_Encoder(nn.Module): 
    def __init__(self, vocab_size, embedding_size=128, encoder_output_normalized=False): 
        super(Text_Encoder, self).__init__() 
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.encoder_output_normalized = encoder_output_normalized

        # Using CNNRNNTEXT Encoder 
        self.emb = nn.Embedding(self.vocab_size, self.embedding_size) # should we initialize the weights using uniform[-1, 1]
        self.cnn = nn.Sequential(
            Conv1d_Pdmd(128, 128, 3, padding_mode='same'), # bz x seq_len x embed_size ->  bz x 128 x ??
            nn.ReLU(),
            Conv1d_Pdmd(128, 128, 3, padding_mode='same'),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            Conv1d_Pdmd(128, 256, 3, padding_mode='same'),
            nn.ReLU(),
            Conv1d_Pdmd(256, 256, 3, padding_mode='same'),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )  

        self.rnn = nn.GRU(input_size=256, hidden_size=256)
        self.fc5 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.fc6 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )

    def compute_sequence_length(self, caption_batch):
        """
        caption_batch: Wrapped in a variable, size: B x C batch, B is batch size, C is max caption lengths, 0 means padding
            a non-zero positive value indicates a word index 
        return:
            seq_length_variable: variable of size batch_size representing the length of each caption in 
            in the current batch 
        """
        seq_length_variable = torch.gt(caption_batch, 0).sum(dim=1)
        seq_length_variable = seq_length_variable.long()
        return seq_length_variable

    def forward(self, caption_batch):
        """
        caption_batch: raw caption batch, size: bz x max_seq_len 
        """
        # calculate length for each caption, and store it in the tensor
        
        max_seq_len =  caption_batch.size(1)
        seq_len_tensor = self.compute_sequence_length(caption_batch)
        # forward caption_batch to embedding layer  
        embedding_batch = self.emb(caption_batch) # bz x seq_len x emb_size 

        x = embedding_batch.permute(0, 2, 1) # bz x emb_size x seq_len
        x_emb = self.cnn(x)
        x_emb = x_emb.permute(0, 2, 1) # bz x seq_len x emb_size 
        x_emb = x_emb.permute(1, 0, 2) # seq_len x bz x emb_size 
        # GRU 
        # before every call lets call flatten the paramters 
        # currently, cannot call it by using DistributedDataParallel
        # self.rnn.flatten_parameters()

        output_all, hidden_T = self.rnn(x_emb, None) # output: seq_len x bz x emb_size 
        
        # take last time-step output 
        # max_seq_len x bz x emb_size
        
        masks = (seq_len_tensor-1).unsqueeze(0).unsqueeze(2).expand(max_seq_len, output_all.size(1), output_all.size(2))
        output = output_all.gather(0, masks)[0]

        output = self.fc5(output) 
        output = self.fc6(output)  # bz x 128

        if self.encoder_output_normalized:
            output = F.normalize(output, p=2, dim=1)

        return output 


####################################################################
## Tensorflow style padding 
####################################################################
def pad_size_3d(ks, mode):

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
    def __init__(self, D_in, D_out, ks=(3, 3, 3), st=(1, 1, 1), pd=0, padding_mode=None, bias=True): 
        super(Conv3d_Pdmd, self).__init__()
        # if padding_mode is specified to be 'same' or 'valid', or 'full'
        # then padding size is computed by ks, here we donot specify pd 
        if padding_mode: 
            pd = pad_size_3d(ks, padding_mode) 

        self.conv = nn.Conv3d(D_in, D_out, ks, st, pd, bias=bias) 

        # initialize 
        # init.kamming_normal(self.conv.weight)
        # if bias: 
        #    init.constant(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x) 

class Shape_Encoder(nn.Module): 
    def __init__(self, num_classes=4, encoder_output_normalized=False):
        super(Shape_Encoder, self).__init__()
        self.encoder_output_normalized = encoder_output_normalized

        self.cnn3d = nn.Sequential(
            Conv3d_Pdmd(4, 64, ks=(3, 3, 3), st=(2, 2, 2), padding_mode='same'), 
            nn.BatchNorm3d(64), 
            nn.ReLU(True),

            Conv3d_Pdmd(64, 128, ks=(3, 3, 3), st=(2, 2, 2), padding_mode='same'), 
            nn.BatchNorm3d(128), 
            nn.ReLU(True),

            Conv3d_Pdmd(128, 256, ks=(3, 3, 3), st=(2, 2, 2), padding_mode='same'), 
            nn.BatchNorm3d(256), 
            nn.ReLU(True),
            )

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.cnn3d(x) # bz x num_features x H x L x W
        # average pooling 
        out = F.avg_pool3d(x, x.size()[2:])
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        encoder_output = out 
        # we normalize ??
        if self.encoder_output_normalized:
            encoder_output = F.normalize(encoder_output, p=2, dim=1)

        out = self.fc2(out)
        prob = F.softmax(out, dim=1)

        output_dict = {
            'logits': out,
            'probabilities': prob,
            'encoder_output': encoder_output
        }

        return output_dict


if __name__ == '__main__':

    debug_here()
    print('testing text encoder')
    model = Text_Encoder(vocab_size=1000, embedding_size=128)
    caption_batch = Variable(torch.LongTensor([[1, 2, 3, 4,5,6,0, 0], [1, 2, 3, 4,0,0,0, 0], 
                                        [1, 2, 3, 4,5,6,3, 0], [1, 2, 3, 0,0,0,0, 0]]))


    output = model(caption_batch)
    
    print('testing shape encoder') 
    model = Shape_Encoder(num_classes=4)
    inputs = Variable(torch.rand(8, 4, 32, 32, 32)) 

    output_dict = model(inputs)



    print('done')

