import argparse 
import numpy as np 
import os 

from lib.data_process import make_data_processes, kill_processes 
import lib.utils as utils 
import models 

from IPython.core.debugger import Tracer

debug_here = Tracer() 

parser = argparse.ArgumentParser(description='main text2voxel train/test file')
parser.add_argument('--dataset', dest='dataset',help='dataset', default='primitives', type=str)

parser.add_argument('--primitives_all_splits_data_path', help='all dataset path ', default='', type=str)
parser.add_argument('--primitives_train_data_path', help='train dataset path ', default='', type=str)
parser.add_argument('--primitives_val_data_path', help='val dataset path ', default='', type=str)
parser.add_argument('--primitives_test_data_path', help='test dataset path ', default='', type=str)

parser.add_argument('--primitive_metric_embeddings_train', help='train embedding path ', default='', type=str)
parser.add_argument('--primitive_metric_embeddings_val', help='val embedding path ', default='', type=str)
parser.add_argument('--primitive_metric_embeddings_test', help='test embedding path ', default='', type=str)

parser.add_argument('--probablematic_nrrd_path', help='probablematic_nrrd_path', default=None, type=str)


parser.add_argument('--batch_size', dest='batch_size', help='batch size', default=8,type=int)
parser.add_argument('--queue_capacity', dest='queue_capacity', help='queue_capacity', default=4,type=int)
parser.add_argument('--num_workers', dest='num_workers', help='num_workers', default=4,type=int)

parser.add_argument('--val_split', dest='split', help='split for val/test (train, val, test)', default=None, type=str)

parser.add_argument('--synth_embedding', dest='synth_embedding', action='store_true')
parser.add_argument('--text_encoder', dest='text_encoder', help='train/test on text encoder', action='store_true')
parser.add_argument('--train_augment_max', dest='train_augment_max', help='train_augment_max', default=10,type=int)

parser.add_argument('--classifier', dest='classifier', help='train/test on classifier', action='store_true')
parser.add_argument('--reed_classifier', dest='reed_classifier', action='store_true')

opts = parser.parse_args()

debug_here() 

def get_inputs_dict(opts):
    """
    Gets the input dict for the current model and dataset.
    """
    if opts.dataset == 'shapenet':
        pass 
        # if (args.text_encoder is True) or (args.end2end is True) or (args.classifier is True):
        #    inputs_dict = utils.open_pickle(cfg.DIR.TRAIN_DATA_PATH)
        #    val_inputs_dict = utils.open_pickle(cfg.DIR.VAL_DATA_PATH)
        #    test_inputs_dict = utils.open_pickle(cfg.DIR.TEST_DATA_PATH)
        # else:  # Learned embeddings
        #    inputs_dict = utils.open_pickle(cfg.DIR.SHAPENET_METRIC_EMBEDDINGS_TRAIN)
        #    val_inputs_dict = utils.open_pickle(cfg.DIR.SHAPENET_METRIC_EMBEDDINGS_VAL)
        #    test_inputs_dict = utils.open_pickle(cfg.DIR.SHAPENET_METRIC_EMBEDDINGS_TEST)
    
    elif opts.dataset == 'primitives':
        # Primitive dataset 
        if ((opts.synth_embedding is True) or (opts.text_encoder is True) or (opts.classifier is True)):
            
            if opts.classifier and not opts.reed_classifier:  # Train on all splits for classifier
                # tf.logging.info('Using all (train/val/test) splits for training')
                # logging using all (trian/val/test) splits for training 
                inputs_dict = utils.open_pickle(opts.primitives_all_splits_data_path)
            else:
                # tf.logging.info('Using train split only for training')
                inputs_dict = utils.open_pickle(opts.primitives_train_data_path)
            val_inputs_dict = utils.open_pickle(opts.primitives_val_data_path)
            test_inputs_dict = utils.open_pickle(opts.primitives_test_data_path)
        else:  # Learned embeddings
            inputs_dict = utils.open_pickle(opts.primitive_metric_embeddings_train)
            val_inputs_dict = utils.open_pickle(opts.primitive_metric_embeddings_val)
            test_inputs_dict = utils.open_pickle(opts.primitive_metric_embeddings_test)
    else:
        raise ValueError('Please use a valid dataset (shapenet, primitives).')

    # Select the validation/test split
    if opts.val_split == 'train':
        val_inputs_dict = inputs_dict
    elif (opts.val_split == 'val') or (opts.split is None):
        val_inputs_dict = val_inputs_dict
    elif args.val_split == 'test':
        val_inputs_dict = test_inputs_dict
    else:
        raise ValueError('Please select a valid split (train, val, test).')

    print('Validation/testing on {} split.'.format(opts.val_split))

    return inputs_dict, val_inputs_dict

def train(train_queue, text_encoder, shape_encoder, shape_generator, shape_critic, criterion, 
            Optimizer_text_enc, Optimizer_shape_enc, Optimizer_shape_gen, Optimizer_shape_crt, epoch, opts):
    pass  
    



def main():
    global opts
    opts.max_epoch = 200 
    print('-------------building network--------------')
    network_class = models.load_model('CWGAN1')

    text_encoder = 
    shape_encoder = 
    shape_generator = 
    shape_critic = 
    
    ###########################################################
    # Dataset and Make Data Loading Process 
    ###########################################################
    inputs_dict, val_inputs_dict = get_inputs_dict(opts)
    # Prefetching data processes 
    # Create worker and data queue for data processing. For training data, use
    # multiple processes to speed up the loading. For validation data, use 1
    # since the queue will be popped every TRAIN.NUM_VALIDATION_ITERATIONS.
    # set up data queue and start enqueue
    np.random.seed(123) 
    data_process_for_class = models.get_data_process_pairs('CWGAN1', is_training=True, opts)
    val_data_process_for_class = models.get_data_process_pairs('CWGAN1', is_training=False, opts) 

    is_training = True 
    if is_training:
        global train_queue, train_processes 
        global val_queue, val_processes 
        train_queue = Queue(opts.queue_capacity)
        train_processes = make_data_processes(data_process_for_class, train_queue, inputs_dict, opts.num_workers, repeat=True)
        
        val_queue = Queue(opts.queue_capacity)
        val_processes = make_data_processes(val_data_process_for_class, val_queue, val_inputs_dict, 1, repeat=True) 
    else: 
        global test_queue, test_processes 
        test_inputs_dict = val_inputs_dict
        test_queue = Queue(opts.queue_capacity) 
        test_processes = make_data_processes(val_data_process_for_class, test_queue, test_inputs_dict, 1, repeat=False) 



    ################################################################################
    ## we begin to train our network 
    ################################################################################
    for epoch in range(opts.max_epoch):
        # train(train_queue, text_encoder, shape_encoder, shape_generator, shape_critic, criterion, 
        #    Optimizer_text_enc, Optimizer_shape_enc, Optimizer_shape_gen, Optimizer_shape_crt, epoch, opts): 

    pass 



if __name__ == '__main__':
    main() 
