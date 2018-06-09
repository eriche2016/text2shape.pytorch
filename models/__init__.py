from models.cwgan_models import Text2ShapeGenerator1, Text2ShapeDiscriminator2
from models.classifier_models import Classifier1, Classifier128
from models.lba_models import Text_Encoder, Shape_Encoder

# from lib.data_process_encoder import (CaptionDataProcess, CaptionDataProcessTestPhase,
#                                      LBADataProcess, LBADataProcessTestPhase)
from lib.data_process_encoder import (LBADataProcess, LBADataProcessTestPhase)
from lib.data_process_gan import (GANDataProcess, GANDataProcessTestPhase,
                                  CWGANMetricEmbeddingDataProcess)

# from lib.data_process_classifier import ShapeClassifierDataProcess


MODELS = {
    'CWGAN1': {'G': Text2ShapeGenerator1, 'D': Text2ShapeDiscriminator2}, 
    'LBA1': {'Text_Encoder': Text_Encoder, 'Shape_Encoder': Shape_Encoder},
    'Classifier1': Classifier1,
    'Classifier128': Classifier128,
}


def get_data_process_pairs(NetClass, opts, is_training=True):
    """Returns the DataProcess class corresponding to the input NetClass.
    Args:
        NetClass: The network class.
        is_training: Boolean flag indicating whether the network is training or not.
    Returns:
        data_process_class: The DataProcess (sub)class corresponding with the input NetClass.
    """
    CWGANProcess = CWGANMetricEmbeddingDataProcess

    # for data proccess for LBA training 
    if is_training: # if we are in training 
        LBAProcess = LBADataProcess
    else: 
        if opts.LBA_test_mode is None: 
            LBAProcess = LBADataProcess
        else: 
            LBAProcess = LBADataProcessTestPhase

    # one type for shapenet, another type for primitives 
    if opts.shapenet_ct_classifier is True: 
        ClassifierProcess = ShapeClassifierDataProcess 
    else: 
        ClassifierProcess = GANDataProcessTestPhase
    
    # use which data process 
    DATA_PROCESS_PAIRS = {
        'CWGAN1': CWGANProcess,
        'Classifier1': ClassifierProcess,
        'Classifier128': ClassifierProcess, 
        'LBA1': LBAProcess
    }

    return DATA_PROCESS_PAIRS[NetClass]


def load_model(name):
    """Creates and returns an instance of the model given its class name.
    Args:
        name: Name of the model (e.g. 'CWGAN1').
    Returns:
        NetClass: The network class corresponding with the input name.
    """
    model_dict = MODELS

    if name not in model_dict:  # invalid model name
        print('Invalid model:', name)

        # print a list of valid model names
        print('Options are:')
        for model in all_models:
            print('\t* {}'.format(model.__name__))
        raise ValueError('Please select a valid model.')

    NetClass = model_dict[name]

    return NetClass
