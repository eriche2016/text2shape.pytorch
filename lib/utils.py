import os 
import nrrd 
import numpy as np
import pickle
import json  
import time 
import collections
import datetime 
import torch 
from lib.render import render_model_id


import pdb 

# read nrrd data 
def read_nrrd(nrrd_filename):
    """
    Reads an NRRD file and returns an RGBA tensor 
    Args: 
        nrrd_filename: filename of nrrd file 
    Returns: 
        voxel tensor: 4-dimensional voxel tensor with 4 channels (RGBA) where the alpha channel 
            is the last channel(aka vx[:, :, :, 3]).
    """
    nrrd_tensor, options = nrrd.read(nrrd_filename)
    assert nrrd_tensor.ndim == 4 

    # convert to float [0,1]
    voxel_tensor = nrrd_tensor.astype(np.float32) / 255 
    # Move channel dimension to last dimensions 
    voxel_tensor = np.rollaxis(voxel_tensor, 0, 4) 

    # Make the model stand up straight by swapping axes
    voxel_tensor = np.swapaxes(voxel_tensor, 0, 1) 
    voxel_tensor = np.swapaxes(voxel_tensor, 0, 2) 

    return voxel_tensor
# write nrrd 
def write_one_voxel2nrrd(voxel_tensor, filename):
    """
    Converts binvox tensor to NRRD (RGBA) format and writes it if a filename is provided.
    Example usage:
        voxel_tensor = np.load('text2voxel/output/tmp/ckpt-10500/0000_voxel_tensor_output.npy')
        _ = nrrd_rw.write_nrrd(voxel_tensor, filename='./models_checkpoint/test_nrrd.nrrd')
    Args:
        voxel_tensor: A tensor representing the binary voxels. Values can range from 0 to 1, and
            they will be properly scaled. The format is [height, width, depth, channels].
        filename: Filename that the NRRD will be written to.
    Writes:
        nrrd_tensor: An RGBA tensor where the channel dimension (RGBA) comes first
            (channels, height, width, depth).
    """
    if voxel_tensor.ndim == 3:  # Add a channel if there is no channel dimension
        voxel_tensor = voxel_tensor[np.newaxis, :]
    elif voxel_tensor.ndim == 4:  # Roll axes so order is (channel, height, width, depth) (not sure if (h, w, d))
        voxel_tensor = np.rollaxis(voxel_tensor, 3)
    else:
        raise ValueError('Voxel tensor must have 3 or 4 dimensions.')

    # Convert voxel_tensor to uint8
    voxel_tensor = (voxel_tensor * 255).astype(np.uint8)

    if voxel_tensor.shape[0] == 1:  # Add channels if voxel_tensor is a binvox tensor
        nrrd_tensor_slice = voxel_tensor
        nrrd_tensor = np.vstack([nrrd_tensor_slice] * 4)
        nrrd_tensor[:3, :, :, :] = 128  # Make voxels gray
        nrrd_tensor = nrrd_tensor.astype(np.uint8)
    elif voxel_tensor.shape[0] == 4:
        nrrd_tensor = voxel_tensor
    elif voxel_tensor.shape[0] != 4:
        raise ValueError('Voxel tensor must be single-channel or 4-channel')

    nrrd.write(filename, nrrd_tensor)


def get_voxel_file(category, model_id, opts):
    """
    get the voxel absolute filepath for the model specified by category and model_id 
    Args: 
        category: category of the model as a string , e.g., '03001627'
        model_id: model id of the model as a string, e.g., '587ee5822bb56bd07b11ae648ea92233'
    Returns: 
        voxel_filepath: Filepath of the binvox file corresponding to category and model_id 
    """
    if opts.dataset == 'shapenet': # shapenet dataset 
        return opts.data_dir % (model_id, model_id) 
    elif opts.dataset == 'primitives': # primitives dataset
        return opts.data_dir % (category, model_id) 
    else: 
        raise ValueError('please use a valid dataset (shapenet, primitives)')

def load_voxel(category, model_id, opts): 
    """
    Loads the voxel tensor given the model category and model ID 
    Args: 
        category: model category
        model_id: model id 
    Returns: 
        voxel tensor of shape (height x width x depth x channels) 
    """
    voxel_fn = get_voxel_file(category, model_id, opts) 
    voxel_tensor = read_nrrd(voxel_fn) 
    return voxel_tensor 



def augment_voxel_tensor(voxel_tensor, max_noise=10):
    """
    Arguments the RGB values of the voxel tensor. The RGB channelss are perturbed by the same single
    noise value, and the noise is sampled from a uniform distribution.
    Args: 
        voxel_tensor: a single voxel tensor 
        max_noise: Integer representing max noise range. We will perform voxel_value + max_noise to 
        augment the voxel tensor, where voxel_value and max_noise are [0, 255].
    Returns: 
        augmented_voxel_tensor: voxel tensor after the data augmentation
    """
    augmented_voxel_tensor = np.copy(voxel_tensor) # do nothing if binvox 
    if (voxel_tensor.ndim == 4) and (voxel_tensor.shape[3] != 1) and (max_noise > 0):
        noise_val = float(np.random.randint(-max_noise, high=(max_noise + 1))) / 255
        augmented_voxel_tensor[:, :, :, :3] += noise_val
        augmented_voxel_tensor = np.clip(augmented_voxel_tensor, 0., 1.)
    return augmented_voxel_tensor

def rescale_voxel_tensor(voxel_tensor):
    """Rescales all values (RGBA) in the voxel tensor from [0, 1] to [-1, 1].
    Args:
        voxel_tensor: A single voxel tensor.
    Returns:
        rescaled_voxel_tensor: A single voxel tensor after rescaling.
    """
    rescaled_voxel_tensor = voxel_tensor * 2. - 1.
    return rescaled_voxel_tensor

def open_pickle(pickle_file):
    """Open a pickle file and return its contents.
    """
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)
    return pickle_data

def convert_idx_to_words(idx_to_word, data_list):
    """Converts each sentence/caption in the data_list using the idx_to_word dict.
    Args:
        idx_to_word: A dictionary mapping word indices (keys) in string format (?) to words.
        data_list: A list of dictionaries. Each dictionary contains a 'raw_embedding' field (among
            other fields) that is a list of word indices.
    Returns:
        sentences: A list of sentences (strings).
    """
    sentences = []
    for idx, cur_dict in enumerate(data_list):
        sentences.append(('%04d  ' % idx) + ' '.join([idx_to_word[str(word_idx)]
                         for word_idx in cur_dict['raw_caption_embedding']
                         if word_idx != 0]))

    return sentences
    

def print_sentences(json_path, data_list):
    # Opens the processed captions generated from tools/preprocess_captions.py
    inputs_list = json.load(open(json_path, 'r'))
    idx_to_word = inputs_list['idx_to_word']

    if isinstance(data_list, list):
        sentences = convert_idx_to_words(idx_to_word, data_list)
    elif isinstance(data_list, np.ndarray):
        sentences = []
        for idx in range(data_list.shape[0]):
            sentences.append(('%04d  ' % idx) + ' '.join([idx_to_word[str(word_idx)]
                             for word_idx in data_list[idx, :] if word_idx != 0]))

    for sentence in sentences:
        print(sentence + '\n')


class Timer(object):
    """A simple timer.
    """

    def __init__(self):
        self.reset()

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def reset(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.


def categorylist2labellist(category_list_batch, category2label, opts): 
    """
    for primitive datasets, a batch data with category list:
        ['torus-cyan-h100-r20', 'torus-cyan-h100-r20', 'pyramid-orange-h50-r50', 
        'pyramid-orange-h50-r50', 'pyramid-yellow-h50-r100', 'pyramid-yellow-h50-r100', 
        'cone-purple-h50-r50', 'cone-purple-h50-r50']
    We convert it to be: 

    """
    if opts.dataset == 'shapenet':
        shape_labels = [category2label[cat] for cat in category_list_batch]
        if len(shape_labels) > opts.batch_size: # TST, MM 
            shape_label_batch = np.asarray(shape_labels[::opts.LBA_n_captions_per_model])
        else: # STS mode, validation 
            shape_label_batch = np.asarray(shape_labels) 
        return torch.from_numpy(shape_label_batch)
    elif opts.dataset == 'primitives':
        shape_labels = [category2label[cat] for cat in category_list_batch
                            for _ in range(opts.LBA_n_primitive_shapes_per_category)]

        if opts.LBA_model_type == 'TST' or opts.LBA_model_type == 'MM':
            shape_label_batch = np.asarray(shape_labels[::opts.LBA_n_captions_per_model])
        elif opts.LBA_model_type == 'STS': # STS mode, validation 
            # test_queue??? is false
            if opts.test_or_val_phase: # test or val phase 
                shape_label_batch = np.asarray(shape_labels)[::opts.LBA_n_primitive_shapes_per_category]
            else:  # we are in training phase 
                shape_label_batch = np.asarray(shape_labels)

        return torch.from_numpy(shape_label_batch)
    else: 
        raise ValueError('Please select a vlid dataset.')




def consolidate_caption_tuples(minibatch_list, outputs_list, opts, embedding_type='text'):
    """
    From a list of tuples which each have the form: 
    (caption, category, model_id, caption_embedding) 
    """
    caption_tuples = []
    seen_text = []
    seen_shapes = []

    for minibatch, outputs in zip(minibatch_list, outputs_list):
        captions_tensor = minibatch['raw_embedding_batch']
        category_list = minibatch['category_list']
        model_list = minibatch['model_list']
        for i in range(captions_tensor.shape[0]):
            if embedding_type == 'shape':
                caption = None
            else:
                caption = captions_tensor[i]
            
            if opts.LBA_model_type == 'STS':    
                category = category_list[int(np.floor(i/2))]
            else: 
                category = category_list[i]
            
            if opts.LBA_model_type == 'STS': 
                model_id = model_list[int(np.floor(i/2))]
            else: 
                model_id = model_list[i]
                
            if embedding_type == 'text':
                caption_embedding_as_tuple = tuple(caption.tolist())
                if not opts.test_all_tuples and (caption_embedding_as_tuple in seen_text):
                    continue
                else:
                    # get caption embedding at index i from the outputs dict 
                    if i is not None: 
                        caption_embedding = outputs['text_encoder'][i] 
                    else: 
                        caption_embedding = outputs['text_encoder']

                    seen_text.append(caption_embedding_as_tuple)
            elif embedding_type == 'shape': 
                if not opts.test_all_tuples and (model_id in seen_shapes):
                    continue
                else:
                    ### get the shape embedding at index i from the outputs dict. This is None for the generic text
                    ### text encoder (which does not learn shape embeddings) 
                    # pdb.set_trace() 
                    if i is not None: 
                        caption_embedding = outputs['shape_encoder'][i] 
                    else: 
                        caption_embedding = outputs['shape_encoder'] 
                    seen_shapes.append(model_id)
            else:
                return ValueError('Please use a valid embedding type (text or shape).')
            
            caption_tuple = (caption, category, model_id, caption_embedding)
            caption_tuples.append(caption_tuple)

    return caption_tuples



#######################################################
## only used for val phase for text only 
#######################################################
def val_phase_text_minibatch_generator(val_inputs_dict, opts):
        """Return a minibatch generator for the val/test phase for TEXT only.
        """
        # Modify self.caption_tuples so it does not contain multiple instances of the same caption
        new_tuples = []
        seen_captions = []
        for cur_tup in val_inputs_dict['caption_tuples']:
            cur_caption = tuple(cur_tup[0].tolist())
            if cur_caption not in seen_captions:
                seen_captions.append(cur_caption)
                new_tuples.append(cur_tup)
        caption_tuples = new_tuples

        # Collect all captions in the validation set
        raw_caption_list = [tup[0] for tup in caption_tuples]
        category_list = [tup[1] for tup in caption_tuples]
        model_list = [tup[2] for tup in caption_tuples]
        caption_list = raw_caption_list

        vx_tensor_shape = [4, opts.voxel_size, opts.voxel_size, opts.voxel_size]
        zeros_tensor = np.zeros([opts.batch_size, *vx_tensor_shape]).astype(np.float32)
        # if opts.batch_size=8, tensor([ 0,  1,  2,  3,  4,  5,  6,  7])
        caption_label_batch = np.asarray(list(range(opts.batch_size)))
        n_captions = len(caption_list)
        n_loop_captions = n_captions - (n_captions % opts.batch_size)
        print('number of captions: {0}'.format(n_captions))
        print('number of captions to loop through for validation: {0}'.format(n_loop_captions)) 
        print('number of batches to loop through for validation: {0}'.format(n_loop_captions/opts.batch_size))
  
        for start in range(0, n_loop_captions, opts.batch_size):
            captions = caption_list[start:(start + opts.batch_size)]
            minibatch = {
                'raw_embedding_batch': np.asarray(captions),
                'voxel_tensor_batch': zeros_tensor,
                'caption_label_batch': caption_label_batch,
                'category_list': category_list[start:(start + opts.batch_size)],
                'model_list': model_list[start:(start + opts.batch_size)]
            }
            yield minibatch

##########################################################################
## For evaluation purpose, copied from kechen 
##########################################################################
def construct_embeddings_matrix(dataset, embeddings_dict, model_id_to_label=None, label_to_model_id=None):
    """
    Construct the embeddings matrix, which is NxD where N is the number of embeddings and D is
    the dimensionality of each embedding.
    Args:
        dataset: String specifying the dataset (e.g. 'synthetic' or 'shapenet')
        embeddings_dict: Dictionary containing the embeddings. It should have keys such as
                the following: ['caption_embedding_tuples', 'dataset_size'].
                caption_embedding_tuples is a list of tuples where each tuple can be decoded like
                so: caption, category, model_id, embedding = caption_tuple.
    """
    assert ( ((model_id_to_label is None) and (label_to_model_id is None)) or
            ((model_id_to_label is not None) and (label_to_model_id is not None)) )

    embedding_sample = embeddings_dict['caption_embedding_tuples'][0][3]
    embedding_dim = embedding_sample.shape[0]
    num_embeddings = embeddings_dict['dataset_size']
    if (dataset == 'shapenet') and (num_embeddings > 30000):
        raise ValueError('Too many ({}) embeddings. Only use up to 30000.'.format(num_embeddings))
    assert embedding_sample.dim() == 1

    # Print info about embeddings
    print('Number of embeddings:', num_embeddings)
    print('Dimensionality of embedding:', embedding_dim)
    print('Estimated size of embedding matrix (GB):',
          embedding_dim * num_embeddings * 4 / 1024 / 1024 / 1024)
    print('')

    # Create embeddings matrix (n_samples x n_features) and vector of labels
    embeddings_matrix = torch.zeros((num_embeddings, embedding_dim))
    labels = torch.zeros((num_embeddings))

    if (model_id_to_label is None) and (label_to_model_id is None):
        model_id_to_label = {}
        label_to_model_id = {}
        label_counter = 0
        new_dicts = True
    else:
        new_dicts = False

    for idx, caption_tuple in enumerate(embeddings_dict['caption_embedding_tuples']):

        # Parse caption tuple
        caption, category, model_id, embedding = caption_tuple

        # Swap model ID and category depending on dataset
        if dataset == 'primitives':
            tmp = model_id
            model_id = category
            category = tmp

        # Add model ID to dict if it has not already been added
        if new_dicts:
            if model_id not in model_id_to_label:
                model_id_to_label[model_id] = label_counter
                label_to_model_id[label_counter] = model_id
                label_counter += 1

        # Update the embeddings matrix and labels vector
        embeddings_matrix[idx] = embedding
        labels[idx] = model_id_to_label[model_id]

        # Print progress
        if (idx + 1) % 10000 == 0:
            print('Processed {} / {} embeddings'.format(idx + 1, num_embeddings))
    return embeddings_matrix, labels, model_id_to_label, num_embeddings, label_to_model_id

def print_model_id_info(model_id_to_label):
    print('Number of models (or categories if synthetic dataset):', len(model_id_to_label.keys()))
    print('')

    # Look at a few example model IDs
    print('Example model IDs:')
    for i, k in enumerate(model_id_to_label):
        if i < 10:
            print(k)
    print('')


def _compute_nearest_neighbors_cosine(fit_embeddings_matrix, query_embeddings_matrix, 
        n_neighbors, fit_eq_query, range_start=0):
    
    if fit_eq_query is True:
        n_neighbors += 1

    # print('Using unnormalized cosine distance')

    # Argsort method
    # unnormalized_similarities = np.dot(query_embeddings_matrix, fit_embeddings_matrix.T)
    # sort_indices = np.argsort(unnormalized_similarities, axis=1)
    # # return unnormalized_similarities[:, -n_neighbors:], sort_indices[:, -n_neighbors:]
    # indices = sort_indices[:, -n_neighbors:]
    # indices = np.flip(indices, 1)

    # Argpartition method
    # query_embeddings_matrix: 3000 x 128 
    # fit_embeddings_matrix: 17000 x 128
    # resulted unnormalized_similarities: 3000 x 17000
    unnormalized_similarities = np.dot(query_embeddings_matrix, fit_embeddings_matrix.T)
    n_samples = unnormalized_similarities.shape[0]
    ################################################################################################
    # np.argpartition: It returns an array of indices of the same shape as a that
    #   index data along the given axis in partitioned order.
    # kth : int or sequence of ints, Element index to partition by. The k-th element will be in its final sorted position and all smaller elements will
    # be moved before it and all larger elements behind it. The order all elements in the partitions is
    # undefined. If provided with a sequence of k-th it will partition all of them into their sorted 
    # position at once.
    #################################################################################################
    sort_indices = np.argpartition(unnormalized_similarities, -n_neighbors, axis=1)
    # -n_neighbors is in its position, all values bigger than sort_indices[-n_neighbors]
    # is on the right
    indices = sort_indices[:, -n_neighbors:]
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, ...., 29999, .., 2999]
    row_indices = [x for x in range(n_samples) for _ in range(n_neighbors)]
    # take out nearest n_neighbors elements
    yo = unnormalized_similarities[row_indices, indices.flatten()].reshape(n_samples, n_neighbors)
    indices = indices[row_indices, np.argsort(yo, axis=1).flatten()].reshape(n_samples, n_neighbors)
    indices = np.flip(indices, 1)

    if fit_eq_query is True:
        n_neighbors -= 1  # Undo the neighbor increment
        final_indices = np.zeros((indices.shape[0], n_neighbors), dtype=int)
        compare_mat = np.asarray(list(range(range_start, range_start + indices.shape[0]))).reshape(indices.shape[0], 1)
        has_self = np.equal(compare_mat, indices)  # has self as nearest neighbor
        any_result = np.any(has_self, axis=1)
        for row_idx in range(indices.shape[0]):
            if any_result[row_idx]:
                nonzero_idx = np.nonzero(has_self[row_idx, :])
                assert len(nonzero_idx) == 1
                new_row = np.delete(indices[row_idx, :], nonzero_idx[0])
                final_indices[row_idx, :] = new_row
            else:
                final_indices[row_idx, :] = indices[row_idx, :n_neighbors]
        indices = final_indices
    return indices


def compute_nearest_neighbors_cosine(fit_embeddings_matrix, query_embeddings_matrix,
                                     n_neighbors, fit_eq_query):
    print('Using unnormalized cosine distance')
    n_samples = query_embeddings_matrix.shape[0]
    if n_samples > 8000:  # Divide into blocks and execute
        def block_generator(mat, block_size):
            for i in range(0, mat.shape[0], block_size):
                yield mat[i:(i + block_size), :]

        block_size = 3000
        blocks = block_generator(query_embeddings_matrix, block_size)
        indices_list = []
        for cur_block_idx, block in enumerate(blocks):
            print('Nearest neighbors on block {}'.format(cur_block_idx + 1))
            cur_indices = _compute_nearest_neighbors_cosine(fit_embeddings_matrix, block,
                                                            n_neighbors, fit_eq_query,
                                                            range_start=cur_block_idx * block_size)
            indices_list.append(cur_indices)
        indices = np.vstack(indices_list)
        return None, indices
    else:
        return None, _compute_nearest_neighbors_cosine(fit_embeddings_matrix,
                                                       query_embeddings_matrix, n_neighbors,
                                                       fit_eq_query)


def compute_nearest_neighbors(fit_embeddings_matrix, query_embeddings_matrix,
                              n_neighbors, metric='cosine'):
    """Compute nearest neighbors.
    Args:
        fit_embeddings_matrix: NxD matrix
    """
    fit_eq_query = False
    # np.allclose: Returns True if two arrays are element-wise equal within a tolerance (by default, 1e-8).
    # 
    if ((fit_embeddings_matrix.shape == query_embeddings_matrix.shape)
        and np.allclose(fit_embeddings_matrix, query_embeddings_matrix)):
        fit_eq_query = True

    if metric == 'cosine':
        distances, indices = compute_nearest_neighbors_cosine(fit_embeddings_matrix,
                                                              query_embeddings_matrix,
                                                              n_neighbors, fit_eq_query)
    else:
        raise ValueError('Use cosine distance.')
    return distances, indices


def compute_pr_at_k(indices, labels, n_neighbors, num_embeddings, fit_labels=None):
    """Compute precision and recall at k (for k=1 to n_neighbors)
    Args:
        indices: num_embeddings x n_neighbors array with ith entry holding nearest neighbors of
                 query i
        labels: 1-d array with correct class of query
        n_neighbors: number of neighbors to consider
        num_embeddings: number of queries
    """
    if fit_labels is None:
        fit_labels = labels
    num_correct = np.zeros((num_embeddings, n_neighbors))
    rel_score = np.zeros((num_embeddings, n_neighbors))
    label_counter = np.bincount(fit_labels)
    num_relevant = label_counter[labels]
    rel_score_ideal = np.zeros((num_embeddings, n_neighbors))

    # Assumes that self is not included in the nearest neighbors
    for i in range(num_embeddings):
        label = labels[i]  # Correct class of the query
        nearest = indices[i]  # Indices of nearest neighbors
        nearest_classes = [fit_labels[x] for x in nearest]  # Class labels of the nearest neighbors
        # for now binary relevance
        num_relevant_clamped = min(num_relevant[i], n_neighbors)
        rel_score[i] = np.equal(np.asarray(nearest_classes), label)
        rel_score_ideal[i][0:num_relevant_clamped] = 1

        for k in range(n_neighbors):
            # k goes from 0 to n_neighbors-1
            correct_indicator = np.equal(np.asarray(nearest_classes[0:(k + 1)]), label)  # Get true (binary) labels
            num_correct[i, k] = np.sum(correct_indicator)

    # Compute our dcg
    dcg_n = np.exp2(rel_score) - 1
    dcg_d = np.log2(np.arange(1,n_neighbors+1)+1)
    dcg = np.cumsum(dcg_n/dcg_d,axis=1)
    # Compute ideal dcg
    dcg_n_ideal = np.exp2(rel_score_ideal) - 1
    dcg_ideal = np.cumsum(dcg_n_ideal/dcg_d,axis=1)
    # Compute ndcg
    ndcg = dcg / dcg_ideal
    ave_ndcg_at_k = np.sum(ndcg, axis=0) / num_embeddings
    recall_rate_at_k = np.sum(num_correct > 0, axis=0) / num_embeddings
    recall_at_k = np.sum(num_correct/num_relevant[:,None], axis=0) / num_embeddings
    precision_at_k = np.sum(num_correct/np.arange(1,n_neighbors+1), axis=0) / num_embeddings
    #print('recall_at_k shape:', recall_at_k.shape)
    print('     k: precision recall recall_rate ndcg')
    for k in range(n_neighbors):
        print('pr @ {}: {} {} {} {}'.format(k + 1, precision_at_k[k], recall_at_k[k], recall_rate_at_k[k], ave_ndcg_at_k[k]))
    Metrics = collections.namedtuple('Metrics', 'precision recall recall_rate ndcg')
    return Metrics(precision_at_k, recall_at_k, recall_rate_at_k, ave_ndcg_at_k)


def get_nearest_info(indices, labels, label_to_model_id, caption_tuples, idx_to_word):
    """Compute and return the model IDs of the nearest neighbors.
    """
    # Convert labels to model IDs
    query_model_ids = []
    query_sentences = []
    for idx, label in enumerate(labels):
        # query_model_ids.append(label_to_model_id[label])
        query_model_ids.append(caption_tuples[idx][2])
        cur_sentence_as_word_indices = caption_tuples[idx][0]
        if cur_sentence_as_word_indices is None:
            query_sentences.append('None (shape embedding)')
        else:
            # word_idx is tensor of size 0
            query_sentences.append(' '.join([idx_to_word[str(word_idx.item())]
                                            for word_idx in cur_sentence_as_word_indices
                                            if word_idx.item() != 0]))

    # Convert neighbors to model IDs
    nearest_model_ids = []
    nearest_sentences = []
    for row in indices:
        model_ids = []
        sentences = []
        for col in row:
            # model_ids.append(label_to_model_id[labels[col]])
            model_ids.append(caption_tuples[col][2])
            cur_sentence_as_word_indices = caption_tuples[col][0]
            if cur_sentence_as_word_indices is None:
                cur_sentence_as_words = 'None (shape embedding)'
            else:
                cur_sentence_as_words = ' '.join([idx_to_word[str(word_idx.item())]
                                                 for word_idx in cur_sentence_as_word_indices
                                                 if word_idx.item() != 0])
            sentences.append(cur_sentence_as_words)
        nearest_model_ids.append(model_ids)
        nearest_sentences.append(sentences)
    assert len(query_model_ids) == len(nearest_model_ids)
    return query_model_ids, nearest_model_ids, query_sentences, nearest_sentences


def print_nearest_info(query_model_ids, nearest_model_ids, query_sentences, nearest_sentences, opts, 
                       render_dir=None):
    """Print out nearest model IDs for random queries.
    Args:
        labels: 1D array containing the label
    """
    # Make directory for renders
    if render_dir is None:
        render_dir = os.path.join('./models_checkpoint/render/', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(render_dir)

    num_queries = 25
    assert len(nearest_model_ids) > num_queries
    perm = np.random.permutation(len(nearest_model_ids))
    for i in perm[:num_queries]:
        query_model_id = query_model_ids[i]
        nearest = nearest_model_ids[i]

        # Make directory for the query
        cur_render_dir = os.path.join(render_dir, query_model_id + ('-%04d' % i))
        os.makedirs(cur_render_dir)

        with open(os.path.join(cur_render_dir, 'nearest_neighbor_text.txt'), 'w') as f:
            f.write('-------- query {} ----------\n'.format(i))
            f.write('Query: {}\n'.format(query_model_id))
            f.write('Nearest:\n')
            for model_id in nearest:
                f.write('\t{}\n'.format(model_id))
            
            render_model_id([query_model_id] + nearest, opts, out_dir=cur_render_dir, check=False)

            f.write('')
            query_sentence = query_sentences[i]
            f.write('Query: {}\n'.format(query_sentence))
            for sentence in nearest_sentences[i]:
                f.write('\t{}\n'.format(sentence))
            f.write('')

        ids_only_fname = os.path.join(cur_render_dir, 'ids_only.txt')
        with open(ids_only_fname, 'w') as f:
            f.write('{}\n'.format(query_model_id))
            for model_id in nearest:
                f.write('{}\n'.format(model_id))



def compute_metrics(embeddings_dict, opts, metric='cosine', concise=False):
    """Compute all the metrics for the text encoder evaluation.
    """
    # assert len(embeddings_dict['caption_embedding_tuples']) < 10000
    # Dont need two sort steps!! https://stackoverflow.com/questions/1915376/is-pythons-sorted-function-guaranteed-to-be-stable
    # embeddings_dict['caption_embedding_tuples'] = sorted(embeddings_dict['caption_embedding_tuples'], key=lambda tup: tup[2])
    # embeddings_dict['caption_embedding_tuples'] = sorted(embeddings_dict['caption_embedding_tuples'], key=lambda tup: tup[0].tolist())
    (embeddings_matrix, labels, model_id_to_label, num_embeddings, label_to_model_id) = construct_embeddings_matrix(opts.dataset, embeddings_dict)

    print('min embedding val:', torch.min(embeddings_matrix).item())
    print('max embedding val:', torch.max(embeddings_matrix).item())
    print('mean embedding (abs) val:', torch.mean(torch.abs(embeddings_matrix)).item())
    print_model_id_info(model_id_to_label)

    n_neighbors = 20

    # if (num_embeddings > 16000) and (metric == 'cosine'):
    #     print('Too many embeddings for cosine distance. Using L2 distance instead.')
    #     metric = 'minkowski'
    # distances, indices = compute_nearest_neighbors(embeddings_matrix, n_neighbors, metric=metric)

    ##############################################################################################################
    ## in the function, we will use numpy
    ##############################################################################################################
    embeddings_matrix = embeddings_matrix.data.numpy() 
    labels = labels.data.numpy().astype(np.int32) 

    distances, indices = compute_nearest_neighbors(embeddings_matrix, embeddings_matrix, n_neighbors, metric=metric)

    print('Computing precision recall.')
    pr_at_k = compute_pr_at_k(indices, labels, n_neighbors, num_embeddings)
    # plot_pr_curve(pr_at_k)

    # Print some nearest neighbor indexes and labels (for debugging)
    # for i in range(10):
    #     print('Label:', labels[i])
    #     print('Neighbor indices:', indices[i][:5])
    #     print('Neighbors:', [labels[x] for x in indices[i][:5]])

    if concise is False or isinstance(concise, str):
        # Opens the processed captions generated from tools/preprocess_captions.py
        if opts.dataset == 'shapenet':
            json_path = shapenet_json_path
        elif opts.dataset == 'primitives': 
            json_path = opts.primitives_json_path
        else:
            raise ValueError('please use a valid dataset (shapenet, primitives)')
            
        with open(json_path, 'r') as f:
            inputs_list = json.load(f)
        idx_to_word = inputs_list['idx_to_word']

        query_model_ids, nearest_model_ids, query_sentences, nearest_sentences = get_nearest_info(
            indices,
            labels,
            label_to_model_id,
            embeddings_dict['caption_embedding_tuples'],
            idx_to_word,
        )

        out_dir = concise if isinstance(concise, str) else None
        print_nearest_info(query_model_ids, nearest_model_ids, query_sentences, nearest_sentences, opts, 
                           render_dir=out_dir)

    return pr_at_k


############################################################################
##
############################################################################
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            # if param.grad is not None:
            param.grad.data.clamp_(-grad_clip, grad_clip)
