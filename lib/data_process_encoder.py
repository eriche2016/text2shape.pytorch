import numpy as np 
import pickle 
import random 
import torch 
from collections import Counter 

from lib.data_process import DataProcess
from lib.utils import load_voxel
 
class LBADataProcess(DataProcess):
    """
    Data process that returns a raw caption batch and a shape batch 
    """
    def __init__(self, data_queue, data_dict, opts, repeat=True):
        ##################################################################
        ## data_dict: 
        ## keys: 
        ##   'caption_tuples': caption_tuples is
        ##        a list of caption tuples, where each caption tuple is (caption, model_category,
        ##        model_id). e.g., inputs_dict['caption_tuples'] = (array([1, 2, 3, 4, 5, 6, 0, 0], 
        ##                                  dtype=int32), 'cone-purple-h20-r100', 'cone-purple-h20-r100_7.nrrd')
        ##   'caption_matches': a dict where the key is any model ID and the value
        ##   is a list of the indices (ints) of caption tuples that describe the same model ID
        ##   'vocab_size': 
        ##   'max_caption_length': 
        ##   .......
        ##################################################################
        
        self.opts = opts 
        self.caption_tuples = data_dict['caption_tuples']

        if opts.dataset == 'shapenet':
            self.caption_matches = data_dict['caption_matches']
        elif opts.dataset == 'primitives':
            self.caption_matches = data_dict['category_matches']
            self.category2modelid = data_dict['category2modelid']
        else: 
            raise ValueError('please select a valid dataset.')

        self.matches_keys = list(self.caption_matches.keys())
        self.n_captions_per_model = opts.LBA_n_captions_per_model 

        if opts.dataset == 'shapenet':
            self.n_unique_shape_categories = opts.batch_size 
            self.n_models_per_batch = self.n_unique_shape_categories
        elif opts.dataset == 'primitives':
            # assert opts.LBA_n_primitive_shapes_per_category == 2 
            assert opts.batch_size % opts.LBA_n_primitive_shapes_per_category == 0
            self.n_unique_shape_categories = opts.batch_size // opts.LBA_n_primitive_shapes_per_category 
            self.n_models_per_batch = opts.batch_size
        else: 
            raise ValueError('Please select a valid dataset')

        super(LBADataProcess, self).__init__(data_queue, range(len(self.caption_matches)), 
            batch_size=self.n_unique_shape_categories, repeat=repeat)

        self.max_sentence_length = len(self.caption_tuples[0][0])

        lengths = []
        for cur_tup in self.caption_matches.values():
            lengths.append(len(cur_tup))

        counter = Counter(lengths) 
        print('dataset statitics')
        print('--> Format - num captions: num models with num captions')
        print('-->', counter)

        if opts.dataset == 'shapenet' and opts.probablematic_nrrd_path is not None: 
            with open(opts.probablematic_nrrd_path, 'rb') as f: 
                self.bad_model_ids = pickle.load(f) 
        else: 
            self.bad_model_ids = None 

    def is_bad_model_id(self, model_id):
        if self.bad_model_ids is not None: 
            return model_id in self.bad_model_ids 
        else: 
            return False 

    def verify_batch(self, caption_tuples_for_cur_key): 
        """
        simply verify that all caption tuples in the batch correspond to the same category and model id 
        """
        category = caption_tuples_for_cur_key[0][1]
        model_id = caption_tuples_for_cur_key[0][2] 
        for tup in caption_tuples_for_cur_key: 
            assert tup[1] == category
            if self.opts.dataset == 'shapenet':
                assert tup[2] == model_id

        return category, model_id

    def run(self):
        """
        category and model lists dynamically change size depending on whether it is STS or TST mode 
        """
        # run the loop until exit flag is set 
        while not self.exit.is_set() and self.cur < self.num_data: 
            # print('{0}/{1} samples'.format(self.cur, self.num_data))

            # Ensure that the network sees (almost) all the data per epoch 
            db_inds = self.get_next_minibatch() 

            shapes_list = []
            captions_list = []
            category_list = []
            model_id_list = []

            for db_ind in db_inds: # Loop through each selected shape 
                selected_shapes = [] 
                while True: 
                    # cur_key is the model id for shapenet, category for primitives 
                    cur_key = self.matches_keys[db_ind] 
                    caption_idxs = self.caption_matches[cur_key]

                    ## Ensure theat len(caption_idxs) >= self.n_captions_per_model
                    if len(caption_idxs) < self.n_captions_per_model: # until len(caption_idxs) == self.n_captions_per_model
                        db_ind = np.random.randint(self.num_data) # take a random index
                        continue 

                    # randomly sample self.n_captions_per_model captions from caption_idxs
                    selected_caption_idxs = random.sample(caption_idxs, k=self.n_captions_per_model)
                    selected_tuples = [self.caption_tuples[idx] for idx in selected_caption_idxs] 
                    # model id is cur_key  
                    cur_category, cur_model_id = self.verify_batch(selected_tuples) 

                    # select shapes/models 
                    if self.opts.dataset == 'shapenet':
                        selected_model_ids = [cur_model_id]
                    elif self.opts.dataset == 'primitives':
                        category_model_ids = self.category2modelid[cur_category]
                        selected_model_ids = random.sample(category_model_ids, k=self.opts.LBA_n_primitive_shapes_per_category)
                    else: 
                        raise ValueError('Please select a valid dataset') 


                    # append cur_shape to selected_shapes  
                    # for shapenet, selected_model_ids = [cur_model_id]
                    # for primitives, category_model_ids = self.category2modelid[cur_category], and 
                    # we will saample self.LBA_n_primitive_shapes_per_category models for this category
                    for cur_model_id in selected_model_ids: 
                        if self.is_bad_model_id(cur_model_id):
                            db_ind = np.random.randint(self.num_data)
                            continue 
                        try: 
                            cur_shape = load_voxel(cur_category, cur_model_id, self.opts)
                        except FileNotFoundError: 
                            print('Error: cannot find file with the following model id: ', cur_key)
                            db_ind = np.random.randint(self.num_data) 
                            continue 
                        selected_shapes.append(cur_shape)
                    break
                # 每个model有self.n_captions_per_model个captions
                selected_captions = [tup[0] for tup in selected_tuples] 
                captions_list.extend(selected_captions)
                # 每个类（对于shapenet，选择1个），选择LBA_n_primitive_shapes_per_category个model
                for selected_shape in selected_shapes:  
                    shapes_list.append(selected_shape) 

                if self.opts.LBA_model_type == 'STS': 
                    category_list.append(cur_category) 
                    model_id_list.append(cur_model_id) 
                elif self.opts.LBA_model_type == 'TST' or self.opts.LBA_model_type == 'MM': 
                    cur_categories = [cur_category for _ in selected_captions] # 复制label self.n_captions_per_model次
                    cur_model_ids = [cur_model_id for _ in selected_captions] # 复制model_id self.n_captions_per_model次
                    category_list.extend(cur_categories)
                    model_id_list.extend(cur_model_ids) 
                else:
                    raise ValueError('Please select a valid LBA mode') 

            # Length is the number of captions 
            # Index/label indicates which captions comes from the same shape 
            # [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
            label_list = [x for x in range(self.n_unique_shape_categories)
                            for _ in range(self.n_captions_per_model)] 

            batch_captions = np.array(captions_list).astype(np.int32)
            batch_shapes = np.array(shapes_list).astype(np.float32) 
            # convert dim 
            batch_shapes = batch_shapes.transpose((0, 4, 2,3,1)) # bz x 32 x 32 x 32 x 4 -> bz x 4 x 32 x 32 x 32
            batch_label = np.array(label_list).astype(np.int32)  
             
            # item in the batch_data is pytorch Tensor 
            # the following will wait until the queue frees 
            batch_data = {
                "raw_embedding_batch": batch_captions, 
                'voxel_tensor_batch': batch_shapes, 
                'caption_label_batch': batch_label, 
                'category_list':category_list, 
                'model_list':model_id_list, 
            }
            
            # kill_processes will run okay when the item in the batch_data is not tensor
            # batch_data = {
            #    "raw_embedding_batch": batch_captions.numpy(), 
            #    'voxel_tensor_batch': batch_shapes.numpy(), 
            #    'caption_label_batch': batch_label.numpy(), 
            #    'category_list':category_list, 
            #    'model_list':model_id_list, 
            #}

            self.data_queue.put(batch_data, block=True)

class LBADataProcessTestPhase(DataProcess):

    def __init__(self, data_queue, data_dict, opts, repeat=False):
        """Initialize the Data Process. In this Data Process, each batch is composed of batch_size
        captions. We simply sample from the set of all captions, so each caption is only seen once
        (strictly) in each epoch for a given data process.
        Args:
            data_queue:
            data_dict: A dict with keys 'caption_tuples' and 'caption_matches'. caption_tuples is
                a list of caption tuples, where each caption tuple is (caption, model_category,
                model_id). caption_matches is a dict where the key is any model ID and the value
                is a list of the indices (ints) of caption tuples that describe the same model ID.
            repeat: Boolean flag indicating whether to continue adding to the queue after the epoch
                has ended.
        """
        self.opts = opts 
        assert opts.LBA_test_mode is not None
        self.mode = opts.LBA_test_mode

        if opts.dataset == 'shapenet':
            self.caption_matches = data_dict['caption_matches']
            self.caption_tuples = data_dict['caption_tuples']
        elif opts.dataset == 'primitives':
            self.caption_matches = data_dict['modelid_matches']

            ####################################################
            ## based on test mode, we remove duplicate captions and form self.caption_tuples 
            ####################################################
            if (self.mode == 'text'):
                if not opts.test_all_tuples:
                    # Modify self.caption_tuples so it does not contain multiple instances of the same caption
                    new_tuples = []
                    seen_captions = []
                    for cur_tup in data_dict['caption_tuples']:
                        cur_caption = tuple(cur_tup[0].tolist())
                        if cur_caption not in seen_captions:
                            seen_captions.append(cur_caption)
                            new_tuples.append(cur_tup)
                    # new_dataset_size = len(new_tuples)
                    self.caption_tuples = new_tuples
                else:  # opts.test_all_tuples == True
                    self.caption_tuples = data_dict['caption_tuples']

            elif (self.mode == 'shape'): # 或者self.model == 'text' but self.test_all_tuples == True
                assert  opts.test_all_tuples == False 
                self.caption_tuples = data_dict['caption_tuples']
            else:
                raise ValueError('Please select a valid LBA test mode.')
        else:
            raise ValueError('Please select a valid dataset.')

        self.matches_keys = list(self.caption_matches.keys())
        self.max_sentence_length = len(self.caption_tuples[0][0])
        #########################################################
        ##
        #########################################################
        if opts.test_all_tuples:
            # Since we use caption_tuples instead of caption_matches, we need to be in text mode
            assert opts.LBA_test_mode == 'text'

        if (opts.LBA_test_mode == 'text') or opts.test_all_tuples:
            # self.caption_tuples 仅提供数据样本个数
            super(LBADataProcessTestPhase, self).__init__(data_queue, self.caption_tuples,
                                                          batch_size=opts.batch_size,
                                                          repeat=repeat)
        elif opts.LBA_test_mode == 'shape':
            # self.caption_matches 仅提供数据样本个数
            super(LBADataProcessTestPhase, self).__init__(data_queue, self.caption_matches,
                                                          batch_size=opts.batch_size,
                                                          repeat=repeat)
        else:
            raise ValueError('Please enter a valid LBA test mode.')

        if self.iters_per_epoch == 0:
            print('iters per epoch is 0! setting to 1.')
            self.iters_per_epoch = 1

    def run(self):
        # Run the loop until exit flag is set
        while not self.exit.is_set() and self.cur < self.num_data:
            # Ensure that the network sees (almost) all data per epoch
            # print('{0}/{1} samples'.format(self.cur, self.num_data))  
            
            db_inds = self.get_next_minibatch()

            data_list = []
            category_list = []  # categories
            model_list = []  # models
            shapes_list = []

            continue_while_loop = False
            for db_ind in db_inds:
                if self.mode == 'text':
                    caption_tuple = self.caption_tuples[db_ind]
                elif self.mode == 'shape':
                    cur_key = self.matches_keys[db_ind]
                    caption_idxs = self.caption_matches[cur_key]

                    # Pick the first caption tuple in the matches keys list
                    caption_tuple = self.caption_tuples[caption_idxs[0]]
                else:
                    raise ValueError('Please enter a valid LBA test mode')

                cur_category = caption_tuple[1]
                cur_model_id = caption_tuple[2]
                try:
                    cur_shape = load_voxel(cur_category, cur_model_id, self.opts)
                except FileNotFoundError:
                    assert len(db_inds) == 1
                    print('File not found.')
                    print('Category:', cur_category)
                    print('Model ID:', cur_model_id)
                    print('Skipping.')
                    db_ind = np.random.randint(self.num_data)  # Choose new caption
                    continue_while_loop = True
                    break

                data_list.append(caption_tuple[0])  # 0th element is the caption
                category_list.append(cur_category)
                model_list.append(cur_model_id)
                shapes_list.append(cur_shape)

            if continue_while_loop is True:
                continue
            ##################################################
            ##
            ##################################################
            batch_captions = np.array(data_list).astype(np.int32)
            batch_shapes = np.array(shapes_list).astype(np.float32)
            batch_shapes = batch_shapes.transpose((0, 4,2, 3,1)) # bz x 4 x 32 x 32 x 32

            if self.opts.LBA_test_mode == 'text':
                # Length is number of captions
                # Index/label indicates which captions come from the same shape
                if self.opts.dataset == 'shapenet':
                    # Map IDs for each shape
                    ids = {}
                    next_id = 0
                    for model_id in model_list:
                        if model_id not in ids:
                            ids[model_id] = next_id
                            next_id += 1

                    label_list = [ids[model_id] for model_id in model_list]
                    batch_label = np.array(label_list).astype(np.int32)
                elif self.opts.dataset == 'primitives':
                    # Map IDs for each shape
                    ids = {}
                    next_id = 0
                    for category_id in category_list:
                        if category_id not in ids:
                            ids[category_id] = next_id
                            next_id += 1

                    label_list = [ids[category_id] for category_id in category_list]
                    batch_label = np.array(label_list).astype(np.int32)
                else:
                    raise ValueError('Please select a valid dataset.')
            elif self.opts.LBA_test_mode == 'shape':
                batch_label = np.array(range(self.opts.batch_size))
            else:
                raise ValueError('Please select a valid LBA test phase mode.')

            batch_data = {
                'raw_embedding_batch': batch_captions,
                'voxel_tensor_batch': batch_shapes,
                'caption_label_batch': batch_label,
                'category_list': category_list,
                'model_list': model_list,
            }

            # The following will wait until the queue frees
            self.data_queue.put(batch_data, block=True)




