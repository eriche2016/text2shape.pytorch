import argparse 
import random 
import numpy as np 
import os 
from multiprocessing import Queue # to store the fetched data in a queue during training 

import torch 
import torch.nn as nn 
import torch.nn.parallel # for multi-GPU training
import torch.backends.cudnn as cudnn
import torch.optim as optim


from lib.data_process import make_data_processes, kill_processes 
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
parser.add_argument('--learning_rate', help='learning rate', default=None, type=float)
parser.add_argument('--decay_steps', help='decay steps', default=None, type=int)
parser.add_argument('--max_epochs', help='number of epochs', default=200, type=int)
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--gradient_clip', type=float, default=0.01)
parser.add_argument('--checkpoint_folder', default=None, help='check point path')

opts = parser.parse_args()

print('args: ')
print(opts) 

# for primitive dataset, we assert that for each batch, 2 shapes per category are choosed 
assert opts.LBA_n_primitive_shapes_per_category == 2 

if opts.checkpoint_folder is None:
    opts.checkpoint_folder = 'models_checkpoint'

# make dir 
os.system('mkdir {0}'.format(opts.checkpoint_folder))

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

    
def train(train_data_queue, text_model, shape_model, optimizer_text_model, optimizer_shape_model, criterion, epoch, opts): 
    """
    Train for one epoch 
    """
    Train_Timer = utils.Timer()
    Data_Timer = utils.Timer()  

    # training mode 
    text_model.train() 
    shape_model.train() 
    opts.test_or_val_phase = False 
    
    iteration = 0
    while iteration < opts.train_iters_per_epoch:
        # Fetch data 
        Data_Timer.tic() 
        minibatch = train_data_queue.get()  
        Data_Timer.toc() 
        shape_category_batch = minibatch['category_list']
        
        raw_embedding_batch = torch.from_numpy(minibatch['raw_embedding_batch']).long()
        caption_labels_batch = torch.from_numpy(minibatch['caption_label_batch']).long()

        shape_batch = torch.from_numpy(minibatch['voxel_tensor_batch']) 
        # the converting will use opts.test_or_val_phase = False 
        shape_labels_batch = utils.categorylist2labellist(shape_category_batch, opts.category2label_dict, opts)
        shape_labels_batch = shape_labels_batch.float()
        # shift to GPU 
        if opts.cuda: 
            raw_embedding_batch = raw_embedding_batch.cuda() 
            shape_batch = shape_batch.cuda() 
            caption_labels_batch = caption_labels_batch.cuda() 
            shape_labels_batch = shape_labels_batch.cuda() 

        ######################################################
        ## Train for one iteration 
        ######################################################
        Train_Timer.tic() 
        text_encoder_outputs = text_model(raw_embedding_batch)
        shape_encoder_outputs = shape_model(shape_batch)

        # LBA　loss 
        # debug_here() 
        lba_loss, _, _ = criterion['LBA_loss'](text_encoder_outputs, shape_encoder_outputs['encoder_output'], caption_labels_batch)
        metric_loss = criterion['Metric_Loss'](text_encoder_outputs, shape_encoder_outputs['encoder_output'])
        # Backward 
        # see equation (3) in the text2shape paper 
        loss = lba_loss + opts.rho * metric_loss
        optimizer_text_model.zero_grad() 
        optimizer_shape_model.zero_grad() 
        loss.backward() 
        # clipping gradient
        # utils.clip_gradient(optimizer_text_model, 0.01)
        # utils.clip_gradient(optimizer_shape_model, 0.01)
        
        optimizer_text_model.step()
        optimizer_shape_model.step() 

        Train_Timer.toc() 


        if iteration % opts.print_feq == 0: 

            print('loss at iter {0}: {1}'.format(iteration, loss.item()))
            # print('queue size: {0}/{1}'.format(train_data_queue.qsize(), opts.queue_capacity)) 
            # print('data fetch (sec/step): %.2f'%Data_Timer.average_time)
            # print('train step (sec/step): %.2f'%Train_Timer.average_time)
            Train_Timer.reset() 
            Data_Timer.reset() 

        iteration = iteration + 1


def validation(val_data_queue, text_model, shape_model, criterion, epoch, opts):
    """
    Run validation on validation set  
    Forward a series of minibatches 
    """
    # evaluation mode 
    text_model.eval() 
    shape_model.eval() 
    opts.test_or_val_phase = True 
    
    iteration = 0
    shape_minibatch_list = []
    shape_outputs_list = []
    ######################################################################
    ## computing shape embeddings 
    ######################################################################
    print('computing shape embeddings...')
    while iteration < opts.val_iters_per_epoch:
        print('iter {0}/{1}'.format(iteration, opts.val_iters_per_epoch))
        np.random.seed(1234)
        # Fetch data 
        minibatch = val_data_queue.get()  
        raw_embedding_batch = torch.from_numpy(minibatch['raw_embedding_batch']).long()
        caption_labels_batch = torch.from_numpy(minibatch['caption_label_batch']).long()
        shape_category_batch = minibatch['category_list']

        shape_batch = torch.from_numpy(minibatch['voxel_tensor_batch'])
        # will use opts.test_or_val_phase
        shape_labels_batch = utils.categorylist2labellist(shape_category_batch, opts.category2label_dict, opts)
        shape_labels_batch = shape_labels_batch.float()
        
        minibatch_save = {
            "raw_embedding_batch": raw_embedding_batch.data.cpu(),
            'caption_labels_batch': caption_labels_batch.data.cpu(),
            'category_list': shape_category_batch,
            'model_list': minibatch['model_list']
        }

        # shift to GPU 
        if opts.cuda: 
            raw_embedding_batch = raw_embedding_batch.cuda() 
            shape_batch = shape_batch.cuda() 
            caption_labels_batch = caption_labels_batch.cuda() 
            shape_labels_batch = shape_labels_batch.cuda() 

        ######################################################
        ## forward 
        ######################################################
        text_encoder_outputs = text_model(raw_embedding_batch)
        shape_encoder_outputs = shape_model(shape_batch)

        ######################################################
        ## using criterion to get p_aba_val, p_target 
        ######################################################
        lba_loss, P_aba,  P_target = criterion['LBA_loss'](text_encoder_outputs, shape_encoder_outputs['encoder_output'], caption_labels_batch)
        if opts.LBA_NO_LBA is False: 
            P_aba_val = P_aba # ?? to be determined 
            P_target_val = P_target

        outputs_dict = {
            'text_encoder': text_encoder_outputs.data.cpu(), 
            'shape_encoder': shape_encoder_outputs['encoder_output'].data.cpu(),
            'logits': shape_encoder_outputs['logits'].data.cpu(), 
            'prediction': torch.argmax(shape_encoder_outputs['logits'], dim=1).data.cpu(),
            'P_aba': P_aba_val.data.cpu(),
            'P_target': P_target_val.data.cpu() 
        }
        
        shape_outputs_list.append(outputs_dict)
        shape_minibatch_list.append(minibatch_save)  

        ###########################
        ## END ?? 
        ###########################
        iteration = iteration + 1
    #############################################################
    shape_caption_tuples = utils.consolidate_caption_tuples(shape_minibatch_list, shape_outputs_list, opts, embedding_type='shape')
    #############################################################
    ## compute classification acc
    #############################################################
    pass 

    #############################################################
    ## computing text embeddings 
    #############################################################
    print('computing text embeddings...')
    # pdb.set_trace() 
    # we will make the generator  every time before we compute text embeddings
    text_minibatch_generator_val = utils.val_phase_text_minibatch_generator(opts.val_inputs_dict, opts) 
    text_minibatch_list = []
    text_outputs_list = []
    for step, minibatch in enumerate(text_minibatch_generator_val): 
        if step % 25 == 0:
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
        ## shape batch is zero tensors 
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

        ######################################################
        ## forward 
        ######################################################
        text_encoder_outputs = text_model(raw_embedding_batch)
        # actually no need to forward the below 
        shape_encoder_outputs = shape_model(shape_batch)

        outputs_dict = {
            'text_encoder': text_encoder_outputs.data.cpu()
        }

        text_outputs_list.append(outputs_dict)
        text_minibatch_list.append(minibatch_save)  
    
    # pdb.set_trace() 
    text_caption_tuples = utils.consolidate_caption_tuples(text_minibatch_list, text_outputs_list, opts, embedding_type='text')


    all_caption_tuples = shape_caption_tuples + text_caption_tuples 
    # print information 
    print('number of computed shape embeddings for validation: {0}'.format(len(shape_caption_tuples)))
    print('number of computed text embeddings for validation: {0}'.format(len(text_caption_tuples)))
    print('total number of computed embeddings for validation: {0}'.format(len(all_caption_tuples))) 

    all_outputs_dict = {'caption_embedding_tuples': all_caption_tuples, 
                    'dataset_size': len(all_caption_tuples)} 

    ############################################################
    ## evaluations 
    ############################################################
    pr_at_k = utils.compute_metrics(all_outputs_dict, opts, concise=True) 
    assert len(pr_at_k) == 4 
    precision, recall, recall_rate, ndcg = pr_at_k 
    cur_val_acc = precision[4] # precision @ 5

    print('Current validation accuracy(@5):', cur_val_acc)

    ############################################################
    ## maybe we should save the computed outputs 
    ############################################################
    return cur_val_acc 

        
def main():
    global opts
    opts.max_epochs = 1000
    opts.voxel_size = 32 # 4 x 32 x 32 x 32 

    ###########################################################
    # Dataset and Make Data Loading Process 
    ###########################################################
    inputs_dict, val_inputs_dict = get_inputs_dict(opts)

    # map category to label ('box-teal-h20-r20' -> 755)
    opts.category2label_dict = inputs_dict['class_labels']
    assert inputs_dict['vocab_size'] == val_inputs_dict['vocab_size']
    opts.vocab_size = inputs_dict['vocab_size']

    # Prefetching data processes 
    # Create worker and data queue for data processing. For training data, use
    # multiple processes to speed up the loading. For validation data, use 1
    # since the queue will be popped every TRAIN.NUM_VALIDATION_ITERATIONS.
    # set up data queue and start enqueue
    np.random.seed(123) 
    data_process_for_class = models.get_data_process_pairs('LBA1', opts, is_training=True)
    val_data_process_for_class = models.get_data_process_pairs('LBA1', opts, is_training=False) 
    
    is_training = True
    if is_training:
        global train_queue, train_processes 
        global val_queue, val_processes 
        train_queue = Queue(opts.queue_capacity)
        train_processes = make_data_processes(data_process_for_class, train_queue, inputs_dict, opts, repeat=True)
        # set number of iterations for training 
        opts.train_iters_per_epoch = train_processes[0].iters_per_epoch

        val_queue = Queue(opts.queue_capacity)
        # now we set number of workers to be 1 
        opts.num_workers = 1
        val_processes = make_data_processes(val_data_process_for_class, val_queue, val_inputs_dict, opts, repeat=True)
        # set number of iterations for validation 
        opts.val_iters_per_epoch = val_processes[0].iters_per_epoch

        #########################################################
        ## minibatch generator for the val/test phase for TEXT only.
        #########################################################
        opts.val_inputs_dict = val_inputs_dict
        # text_minibatch_generator_val = utils.val_phase_text_minibatch_generator(opts.val_inputs_dict, opts) 

    else: 
        global test_queue, test_processes 
        test_inputs_dict = val_inputs_dict
        test_queue = Queue(opts.queue_capacity) 
        opts.num_workers = 1 
        test_processes = make_data_processes(val_data_process_for_class, test_queue, test_inputs_dict, opts, repeat=False) 
        # set number of iterations for test
        opts.test_iters_per_epoch = test_processes[0].iters_per_epoch


    ###########################################################
    ## build network 
    ###########################################################
    print('-------------building network--------------')
    network_class = models.load_model('LBA1')
    text_encoder = network_class['Text_Encoder'](opts.vocab_size, embedding_size=128, encoder_output_normalized=False) 
    shape_encoder = network_class['Shape_Encoder'](num_classes=opts.num_classes, encoder_output_normalized=False) 

    print('text encoder: ')
    print(text_encoder)
    print('shape encoder: ')
    print(shape_encoder)
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
    ## optimizer 
    ###########################################################
    optimizer_text_encoder = optim.Adam(text_encoder.parameters(), lr=opts.learning_rate) 
    optimizer_shape_encoder = optim.Adam(shape_encoder.parameters(), lr=opts.learning_rate) 

    ################################################################################
    ## we begin to train our network 
    ################################################################################
    # while True: 
    #    caption_batch = train_queue.get()
    #    caption_batch = val_queue.get() 
    best_val_acc = 0 
    for epoch in range(opts.max_epochs):
    
        print('--------epoch {0}/{1}--------'.format(epoch, opts.max_epochs))
        # train for one epoch 
        train(train_queue, text_encoder, shape_encoder, optimizer_text_encoder, optimizer_shape_encoder, criterion, epoch, opts)
        
        # validation for one epoch 
        if epoch % 25 == 0: 
            print('evaluation...')
            cur_val_acc = validation(val_queue, text_encoder, shape_encoder, criterion, epoch, opts)

            if cur_val_acc > best_val_acc: 
                print('current val acc is bigger than previous best val acc, let us checkpointing ...')
                path_checkpoint = '{0}/model_best.pth'.format(opts.checkpoint_folder)
                checkpoint = {}
                if opts.ngpu > 1:
                    checkpoint['text_encoder'] = text_encoder.module.state_dict()
                    checkpoint['shape_encoder'] = shape_encoder.module.state_dict()
                else: 
                    checkpoint['text_encoder'] = text_encoder.state_dict()
                    checkpoint['shape_encoder'] = shape_encoder.state_dict()

                print('save checkpoint to: ')
                print(path_checkpoint)
                torch.save(checkpoint, path_checkpoint)


    #################################################################################
    # Finally, we kill all the processes 
    #################################################################################
    kill_processes(train_queue, train_processes)
    # kill validation process 
    kill_processes(val_queue, val_processes)
    # if there is test process, also kill it 
    if not is_training: 
        kill_processes(test_queue, test_processes)


if __name__ == '__main__':
    main() 
