import time 
import argparse
from lib.data_process_encoder import LBADataProcess
import numpy as np 
import pickle 
from lib.utils import open_pickle, print_sentences
from lib.data_process import kill_processes

from lib.utils import read_nrrd

import pdb 

def test_data_loading():
    # pdb.set_trace()  
    nrrd_filepath = '/home/hxw/project_work_on/shape_research/datasets/text2shape/shapenet_dataset/nrrd_256_filter_div_32_solid/a85e81c62bed11ea7f4e21263a28c500/a85e81c62bed11ea7f4e21263a28c500.nrrd'
    voxel_tensor = read_nrrd(nrrd_filepath)
    print('done')

# test LBA process 
def test_lba_process():
    from multiprocessing import Queue 
    from lib.utils import print_sentences
    parser = argparse.ArgumentParser(description='test data process')
    parser.add_argument('--dataset', dest='dataset',help='dataset', default='shapenet', type=str)
    opts = parser.parse_args()
    opts.batch_size = 8 
    opts.LBA_n_captions_per_model = 5
    opts.synth_embedding = False 
    opts.probablematic_nrrd_path = '/home/hxw/project_work_on/shape_research/datasets/text2shape/shapenet_dataset/shapenet_info/problematic_nrrds_shapenet_unverified_256_filtered_div_with_err_textures.p'
    opts.LBA_model_type = 'STS'
    opts.val_data_path = '/home/hxw/project_work_on/shape_research/datasets/text2shape/shapenet_dataset/shapenet_info/processed_captions_val.p'
    opts.data_dir = '/home/hxw/project_work_on/shape_research/datasets/text2shape/shapenet_dataset/nrrd_256_filter_div_32_solid/%s/%s.nrrd'

    caption_data = open_pickle(opts.val_data_path) 
    data_queue = Queue(3) # 3代表队列中存放的数据个数上线，达到上限，就会发生阻塞，直到队列中的数据被消费掉
    json_path = '/home/hxw/project_work_on/shape_research/datasets/text2shape/shapenet_dataset/shapenet_info/shapenet.json'
    
    pdb.set_trace() 
    data_process = LBADataProcess(data_queue, caption_data, opts, repeat=True)
    data_process.start()
    caption_batch = data_queue.get() 
    category_list = caption_batch['category_list']
    model_list = caption_batch['model_list'] 

    for k, v in caption_batch.items():
        if isinstance(v, list):
            print('key: ', k) 
            print('value length: ', len(v)) 
        elif isinstance(v, np.ndarray): 
            print('key: ', k)
            print('Value shape: ', v.shape)
        else:
            print('Other: ', k) 
    print('') 
    pdb.set_trace()
    """
    for i in range(len(category_list)):
        print('-------%03d------'%i)
        category = category_list[i] 
        model_id = model_list[i] 

        # generate sentencce 
        for j in range(data_process.n_captions_per_model):
            caption_idx = data_process.n_captions_per_model * i + j 
            caption = caption_batch['raw_embedding_batch'][caption_idx] 

            # print('caption:', caption)
            # print('converted caption: ')
            data_list = [{'raw_caption_embedding': caption}] 
            print_sentences(json_path, data_list)
            print('label: ', caption_batch['caption_label_batch'][caption_idx].item()) 

        print('category: ', category) 
        print('model id: ', model_id) 
    """
    pdb.set_trace() 
    
    kill_processes(data_queue, [data_process])


if __name__ == '__main__':
    test_data_loading()
    test_lba_process()
