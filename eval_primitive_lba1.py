import argparse 
import random 
import numpy as np 
import os 
from multiprocessing import Queue # to store the fetched data in a queue during training 

import torch 
import torch.nn as nn 
import torch.nn.parallel # for multi-GPU training
import torch.backends.cudnn as cudnn

from lib.data_process import make_data_processes, kill_processes, get_while_running
import lib.utils as utils 
import lib.custom_losses as loss
import models 

# for debugging
import pdb 
from IPython.core.debugger import Tracer

debug_here = Tracer() 

parser = argparse.ArgumentParser(description='main text2voxel train/test file')
parser.add_argument('--dataset',help='dataset', default='primitives', type=str)
parser.add_argument('--render_tool_dir', help='render_tool_dir ', default='/home/hxw/project_work_on/shape_research/text2shape.pytorch/tools/sstk', type=str)
parser.add_argument('--data_dir', help='data directory ', default='/home/hxw/project_work_on/shape_research/datasets/text2shape/primitives_dataset/raw_primitives_dataset/%s/%s', type=str)
# for shapenet set it to be 
# /home/hxw/project_work_on/shape_research/datasets/text2shape/shapenet_dataset/nrrd_256_filter_div_32_solid/%s/%s.nrrd

parser.add_argument('--LBA_model_type',help='STS, or TST, or MM', default='STS', type=str)
parser.add_argument('--LBA_test_mode', help='LBA test mode (shape, text) - what to input during forward pass', default=None, type=str)
parser.add_argument('--test_all_tuples', action='store_true')
parser.add_argument('--LBA_n_captions_per_model', help='LBA_n_captions_per_model', default=2,type=int)
parser.add_argument('--LBA_cosin_dist', action='store_true')
parser.add_argument('--LBA_unnormalize', action='store_true')
####################################
# Args for primitives
####################################
parser.add_argument('--primitives_json_path', help='primitives dataset json path ', default='/home/hxw/project_work_on/shape_research/datasets/text2shape/primitives_dataset/primitives/primitives.json', type=str)
parser.add_argument('--LBA_n_primitive_shapes_per_category', help='LBA_n_primitive_shapes_per_category', default=2, type=int)
parser.add_argument('--primitives_all_splits_data_path', help='all dataset path ', default='/home/hxw/project_work_on/shape_research/datasets/text2shape/primitives_dataset/primitives/combined_splits.p', type=str)
parser.add_argument('--primitives_train_data_path', help='train dataset path ', default='/home/hxw/project_work_on/shape_research/datasets/text2shape/primitives_dataset/primitives/processed_captions_train.p', type=str)
parser.add_argument('--primitives_val_data_path', help='val dataset path ', default='/home/hxw/project_work_on/shape_research/datasets/text2shape/primitives_dataset/primitives/processed_captions_val.p', type=str)
parser.add_argument('--primitives_test_data_path', help='test dataset path ', default='/home/hxw/project_work_on/shape_research/datasets/text2shape/primitives_dataset/primitives/processed_captions_test.p', type=str)

parser.add_argument('--primitive_metric_embeddings_train', help='train embedding path ', default='', type=str)
parser.add_argument('--primitive_metric_embeddings_val', help='val embedding path ', default='', type=str)
parser.add_argument('--primitive_metric_embeddings_test', help='test embedding path ', default='', type=str)


####################################
# Args for shapenet 
####################################
parser.add_argument('--shapenet_ct_classifier', help='chair/table classifier (sets up for classification)', action='store_true')
parser.add_argument('--shapenet_json_path', help='shapenet dataset json path ', default='/home/hxw/project_work_on/shape_research/datasets/text2shape/shapenet_dataset/shapenet_info/shapenet.json', type=str)
parser.add_argument('--probablematic_nrrd_path', help='probablematic_nrrd_path', default=None, type=str)
# for shapenet, the set it to be 
# /home/hxw/project_work_on/shape_research/datasets/text2shape/shapenet_dataset/shapenet_info/problematic_nrrds_shapenet_unverified_256_filtered_div_with_err_textures.p

parser.add_argument('--batch_size', help='batch size', default=100,type=int)
parser.add_argument('--queue_capacity', help='queue_capacity', default=3,type=int)
parser.add_argument('--num_workers', help='num_workers', default=1,type=int)
parser.add_argument('--print_feq', help='print every iterations', default=1,type=int)
parser.add_argument('--val_split', help='split for val/test (train, val, test)', default=None, type=str)

#######
parser.add_argument('--synth_embedding', action='store_true')
parser.add_argument('--text_encoder', help='train/test on text encoder', action='store_true')
parser.add_argument('--classifier', help='train/test on classifier', action='store_true')
parser.add_argument('--reed_classifier', help='reed classifier', action='store_true')

parser.add_argument('--train_augment_max', help='train_augment_max', default=10,type=int)

############################################
## Training 
#############################################
parser.add_argument('--pretrained_model', help='path to pretrained model', default=None, type=str)
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--checkpoint_folder', default='./models_checkpoint', help='check point path')

opts = parser.parse_args()

print('args: ')
print(opts) 

# for primitive dataset, we assert that for each batch, 2 shapes per category are choosed 
assert opts.LBA_n_primitive_shapes_per_category == 2 

opts.ngpu = int(opts.ngpu)
opts.manualSeed = 123456
if torch.cuda.is_available() and not opts.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
else:
    if opts.ngpu  == 1: 
        print('so we use 1 gpu to training') 

        if opts.cuda:
            torch.cuda.manual_seed(opts.manualSeed)

cudnn.benchmark = True
print("Random Seed: ", opts.manualSeed)
random.seed(opts.manualSeed)
torch.manual_seed(opts.manualSeed)


if opts.dataset == 'primitives':
    opts.num_classes = 756 # for primitives dataset, there are 756 categories in total 
elif opts.dataset == 'shapenet': 
    opts.num_classes = 2 # Chair table classification 
else: 
    raise ValueError('Please use a supported dataset (shapenet, primitives).')


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
                print('using all (train/val/test) splits for training.')
                inputs_dict = utils.open_pickle(opts.primitives_all_splits_data_path)
            else:
                print('training using train split only.')
                inputs_dict = utils.open_pickle(opts.primitives_train_data_path)
            val_inputs_dict = utils.open_pickle(opts.primitives_val_data_path)
            test_inputs_dict = utils.open_pickle(opts.primitives_test_data_path)
        else:  # Learned embeddings
            inputs_dict = utils.open_pickle(opts.primitives_metric_embeddings_train)
            val_inputs_dict = utils.open_pickle(opts.primitives_metric_embeddings_val)
            test_inputs_dict = utils.open_pickle(opts.primitives_metric_embeddings_test)
    else:
        raise ValueError('Please use a valid dataset (shapenet, primitives).')

    # Select the validation/test split
    if opts.val_split == 'train':
        val_split_str = 'train' 
        val_inputs_dict = inputs_dict
    elif (opts.val_split == 'val') or (opts.val_split is None):
        val_split_str = 'val'
        val_inputs_dict = val_inputs_dict
    elif opts.val_split == 'test':
        val_split_str = 'test'
        val_inputs_dict = test_inputs_dict
    else:
        raise ValueError('Please select a valid split (train, val, test).')

    print('Validation/testing on {} split.'.format(val_split_str))

    if opts.dataset == 'shapenet' and opts.shapenet_ct_classifier is True: 
        pass 

    return inputs_dict, val_inputs_dict

def test_phase_minibatch_generator(test_process, test_queue): 
    for step, minibatch in enumerate(get_while_running(test_process,  test_queue)): 
        yield minibatch 

def test(test_process, test_queue, text_model, shape_model, criterion, opts):
    """
    Run validation on validation set  
    Forward a series of minibatches 
    """
    # evaluation mode 
    text_model.eval() 
    shape_model.eval() 
    opts.test_or_val_phase = True 
    
    minibatch_generator = test_phase_minibatch_generator(test_process, test_queue)
    
    iteration = 0
    minibatch_list = []
    outputs_list = []
    # debug_here() 
    for step, minibatch in enumerate(minibatch_generator):
        np.random.seed(1234)
 
        print('step {0}'.format(step))
        minibatch_save = {
            'raw_embedding_batch': torch.from_numpy(minibatch['raw_embedding_batch']), 
            'caption_labels_batch': torch.from_numpy(minibatch['caption_label_batch']), 
            'category_list': minibatch['category_list'], 
            'model_list': minibatch['model_list']
        }

        raw_embedding_batch = torch.from_numpy(minibatch['raw_embedding_batch']).long()
        caption_labels_batch = torch.from_numpy(minibatch['caption_label_batch']).long()
        shape_category_batch = minibatch['category_list']
        #################################################
        ## 
        #################################################
        shape_batch = torch.from_numpy(minibatch['voxel_tensor_batch'])
        # will use opts.test_or_val_phase
        shape_labels_batch = utils.categorylist2labellist(shape_category_batch, opts.category2label_dict, opts)
        shape_labels_batch = shape_labels_batch.float()
        # shift to GPU 
        if opts.cuda: 
            raw_embedding_batch = raw_embedding_batch.cuda() 
            shape_batch = shape_batch.cuda() 
            caption_labels_batch = caption_labels_batch.cuda() 
            shape_labels_batch = shape_labels_batch.cuda() 

        ## forward 
        text_encoder_outputs = text_model(raw_embedding_batch)
        shape_encoder_outputs = shape_model(shape_batch)

        ######################################################
        ## using criterion to get p_aba_val, p_target 
        ######################################################
        lba_loss, P_aba,  P_target = criterion['LBA_loss'](text_encoder_outputs, shape_encoder_outputs['encoder_output'], caption_labels_batch)
        if opts.LBA_NO_LBA is False: 
            P_aba_test = P_aba # ?? to be determined 
            P_target_test = P_target

        outputs_dict = {
            'text_encoder': text_encoder_outputs.data.cpu(), 
            'shape_encoder': shape_encoder_outputs['encoder_output'].data.cpu(),
            'logits': shape_encoder_outputs['logits'].data.cpu(), 
            'prediction': torch.argmax(shape_encoder_outputs['logits'], dim=1).data.cpu(),
            'P_aba': P_aba_test.data.cpu(),
            'P_target': P_target_test.data.cpu() 
        }
        minibatch_list.append(minibatch_save)
        outputs_list.append(outputs_dict)

    #########################################
    ## evalute 
    #########################################
    print('now let us evalute the output...')
    if not (opts.dataset =='primitives' and opts.LBA_test_mode == 'shape'): 
        print('saving text embeddings')
        text_file_name='text_embeddings.t7'
        caption_tuples = utils.consolidate_caption_tuples(minibatch_list, outputs_list, opts, embedding_type='text')
        class_name2labels_id = opts.category2label_dict
        text_dict = {
            'caption_embedding_tuples': caption_tuples, 
            'dataset_size': len(caption_tuples), 
            'class_labels': class_name2labels_id
        }
        #######################################################
        ## text2text retrieval 
        #######################################################
        pr_at_k = utils.compute_metrics(text_dict, opts, concise=True)
        # debug_here() 
        ###########################################
        ## save to output 
        ###########################################
        print('saving outputs.')
        output_path = os.path.join(opts.checkpoint_folder, text_file_name) 
        print('writing text embeddings to {0}'.format(output_path))
        with open(output_path, 'wb') as f:
            torch.save(text_dict, f)


    if not ((opts.dataset == 'primitives') and (opts.LBA_test_mode == 'text')):
            # Save shape embeddings
            print('save shape embeddings..')
            shape_file_name = 'shape_embeddings.t7'
            shape_tuples = utils.consolidate_caption_tuples(minibatch_list, outputs_list, opts, embedding_type='shape')
            class_labels = opts.category2label_dict

            shape_dict = {
                'caption_embedding_tuples': shape_tuples,
                'dataset_size': len(shape_tuples),
                'class_labels': class_labels
            }
            #######################################################
            ## shape2shape retrieval 
            #######################################################
            
            pr_at_k = utils.compute_metrics(shape_dict, opts, concise=True)

            print('saving outputs...')
            output_path = os.path.join(opts.checkpoint_folder, shape_file_name) 
            print('writing shape embeddings to {0}'.format(output_path))
            with open(output_path, 'wb') as f:
                torch.save(shape_dict, f)

    if not opts.dataset == 'primitives':
               # Combine text embeddings and shape embeddings
            print('Combined text and shape embeddings')
            combined_dict = {
                'caption_embedding_tuples': text_dict['caption_embedding_tuples'] + shape_dict['caption_embedding_tuples'],
                'dataset_size': text_dict['dataset_size'] + shape_dict['dataset_size'],
            }
            pr_at_k = utils.compute_metrics(combined_dict, opts, concise=True)
            print('saving output')
            output_path = os.path.join(opts.checkpoint_folder, 'text_and_shape_embeddings.p')
            print('writing text and shape embeddings to {0}'.format(output_path))
            with open(output_path, 'wb') as f:
                torch.save(combined_dict, f)
    
    # debug_here()   
    return pr_at_k
        
def main():
    global opts
    opts.voxel_size = 32 # 4 x 32 x 32 x 32 

    ###########################################################
    # Dataset and Make Data Loading Process 
    ###########################################################
    inputs_dict, test_inputs_dict = get_inputs_dict(opts)

    # map category to label ('box-teal-h20-r20' -> 755)
    opts.category2label_dict = inputs_dict['class_labels']
    assert inputs_dict['vocab_size'] == test_inputs_dict['vocab_size']
    opts.vocab_size = inputs_dict['vocab_size']

    # Prefetching data processes 
    # Create worker and data queue for data processing. For training data, use
    # multiple processes to speed up the loading. For validation data, use 1
    # since the queue will be popped every TRAIN.NUM_VALIDATION_ITERATIONS.
    # set up data queue and start enqueue
    np.random.seed(123) 
    test_data_process_for_class = models.get_data_process_pairs('LBA1', opts, is_training=False) 
    
    global test_queue, test_processes 
    test_queue = Queue(opts.queue_capacity) 
    opts.num_workers = 1 
    test_processes = make_data_processes(test_data_process_for_class, test_queue, test_inputs_dict, opts, repeat=False) 

    ###########################################################
    ## build network, loading pretrained model, shift to GPU 
    ###########################################################
    print('-------------building network--------------')
    network_class = models.load_model('LBA1')
    text_encoder = network_class['Text_Encoder'](opts.vocab_size, embedding_size=128, encoder_output_normalized=False) 
    shape_encoder = network_class['Shape_Encoder'](num_classes=opts.num_classes, encoder_output_normalized=False) 

    print('text encoder: ')
    print(text_encoder)
    print('shape encoder: ')
    print(shape_encoder)

    print('loading checkpoints....')
    if opts.pretrained_model != '':
        print('loading pretrained model from {0}'.format(opts.pretrained_model))
        checkpoint = torch.load(opts.pretrained_model)
        text_encoder.load_state_dict(checkpoint['text_encoder'])
        shape_encoder.load_state_dict(checkpoint['shape_encoder'])
    else: 
        assert ValueError('please input the path to pretrained model.')

    ###########################################################
    ## Training Criterion 
    ###########################################################
    criterion = {}
    opts.LBA_NO_LBA = False 
    if opts.LBA_NO_LBA is False: 
        # by default, we set visit loss weith to be 0.25
        LBA_loss = loss.LBA_Loss(lmbda=0.25, LBA_model_type=opts.LBA_model_type, batch_size=opts.batch_size)
        criterion['LBA_loss'] = LBA_loss

    # classificaiton loss 
    opts.LBA_Classificaiton = False 
    if opts.LBA_Classificaiton is True: 
        pass 

    # metric loss
    opts.LBA_Metric = True 
    if opts.LBA_Metric is True: 
        opts.rho = 1.0 # set opts.rho to be 1.0 for combining LBA_loss and Metric loss 
        Metric_loss = loss.Metric_Loss(opts, LBA_inverted_loss=True, LBA_normalized=False, LBA_max_norm=10.0)
        criterion['Metric_Loss'] = Metric_loss

    ## shift models to cuda 
    if opts.cuda:  
        print('shift model and criterion to GPU .. ')
        text_encoder = text_encoder.cuda() 
        shape_encoder = shape_encoder.cuda() 
        if opts.ngpu > 1:
            text_encoder = nn.DataParallel(text_encoder, device_ids=range(opts.ngpu)) 
            shape_encoder = nn.DataParallel(shape_encoder, device_ids=range(opts.ngpu)) 

        for crit in criterion.values(): 
            crit = crit.cuda() 
    

    ###########################################################
    ## Now we begin to test 
    ###########################################################         
    print('evaluation...')
    pr_at_k = test(test_processes[0], test_queue, text_encoder, shape_encoder, criterion, opts)

    #################################################################################
    # Finally, we kill all the processes 
    #################################################################################
    kill_processes(test_queue, test_processes)



if __name__ == '__main__':
    main() 
