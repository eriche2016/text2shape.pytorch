import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

# from torch.autograd import Variable 

import pdb 

def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)

def pairwise_dot(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: sim_mat is a NxM matrix where sim_mat[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. sim_mat[i,j] = x[i,:] * y[j,:]
    '''
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
    else:
        y_t = torch.transpose(x, 0, 1)
    
    sim_mat = torch.mm(x, y_t)

    return sim_mat

#################################################
## TST goal: for a given description i, we want P_TST(i,j) be unifrom over discription j which 
## are similar to i. Thus we define round-trip loss L_TST as the cross-entropy between P_TST and the
## target uniform distribution. 
#################################################
class softCrossEntropy_v1(nn.Module):
    def __init__(self):
        super(softCrossEntropy_v1, self).__init__()

    def forward(self, inputs, soft_target):
        """
        :param inputs: predictions
        :param target: target labels
        :return: loss
        """
        neg_log_likelihood = -F.log_softmax(inputs, dim=1)
        sample_num, _ = soft_target.shape
        # if not specified dim in torch.sum(), it will sum all elements in the tensor 
        loss = torch.sum(torch.mul(neg_log_likelihood, soft_target))/sample_num

        return loss

class softCrossEntropy_v2(nn.Module):
    def __init__(self):
        super(softCrossEntropy_v2, self).__init__()

    def forward(self, inputs, soft_target):
        """
        :param inputs: predictions
        :param target: target labels
        :return: loss
        """
        # take log function
        # pdb.set_trace()
        if False:  
            inputs = torch.log(inputs + 1e-8)
            neg_likelihood = -F.softmax(inputs, dim=1)
            sample_num, _ = soft_target.shape
            # if not specified dim in torch.sum(), it will sum all elements in the tensor 
            loss = torch.sum(torch.mul(neg_likelihood, soft_target))/sample_num
        else: 
            inputs = torch.log(inputs + 1e-8)
            inputs = F.softmax(inputs, dim=1)
            cross_entropy_loss = soft_target * torch.log(inputs) # + (1 - soft_target) * torch.log(1-inputs) 
            cross_entropy_loss = -1 * cross_entropy_loss 
            loss = torch.mean(torch.sum(cross_entropy_loss, dim=1))

        return loss

###################################################################
## Used to build STS and TST loss 
###################################################################
class Semisup_Loss(nn.Module):
    def __init__(self, lmbda=1, lba_dist_type='standard'):
        """
        semi-supervised classification loss to the model, 
        The loss constist of two terms: "walker" and "visit".

        Args:
        A: [N, emb_size] tensor with supervised embedding vectors.
        B: [M, emb_size] tensor with unsupervised embedding vectors.
        labels : [N] tensor with labels for supervised embeddings.
        walker_weight: Weight coefficient of the "walker" loss.
        visit_weight: Weight coefficient of the "visit" loss.

        Return: 
            return the sum of two losses 
        """
        super(Semisup_Loss, self).__init__() 
        self.lmbda = lmbda 
        self.lba_dist_type = lba_dist_type

        # note nn.CrossEntropyLoss() only support the case when target is categorical value, e.g., 0, 1, ..
        self.cross_entropy = softCrossEntropy_v2() #

    def build_walk_stastistics(self, p_tst, equality_matrix):
        """
        compute walker loss stastistics
        Args: 
            p_tst: N x N matrix, where [j,j] element corresponds to the probability 
            of the round-trip between supervised samples i and j 
            sum of each row of p_tst must be equal to 1
            equality_matrix: N x N matrix, [i,j] == 1 -> samples i and j belong to the same class   
        """
        # Using the sequare root of the correct round trip probability as an estimate of the 
        # current classifier accuracy 
        per_row_accuracy = 1.0 - torch.sum(equality_matrix * p_tst, dim=1) ** 0.5 
        estimate_error = torch.mean(1-per_row_accuracy)
        return estimate_error


    def forward(self, A, B, labels): 
        """
        compute similarity matrix 
        Args: 
            A: size: N x emb_size, tensor with supervised embedding vectors 
            B: size: M x emb_size, tensor with unsupervised embedding vectors  
            labels: size: N, tensor with labels for supervised embeddings 
            for ease of understanding, currently, A -> text_embedding, B -> shape_embedding, labels -> caption_labels
        """ 
        # build target probability distribution matrix based on uniform dist over correct labels 
        # N x N 
        equality_matrix = torch.eq(*[labels.unsqueeze(dim).expand(A.size(0), A.size(0)) for dim in [0, 1]]).type_as(A)
        p_target = torch.div(equality_matrix, torch.sum(equality_matrix, dim=1, keepdim=True))

        if self.lba_dist_type == 'standard':
            M = pairwise_dot(A, B) # N x M, each row i: sim(text_i, shape_1), sim(text_i, shape_2),sim(text_i, shape_3) ...
        elif self.lba_dist_type ==  'mahalanobis': 
            print('to be implemented')
            raise ValueError('currently, not support malanobis distance.')
        else: 
            return ValueError('please select a valid distance type')

        M_t = M.transpose(0, 1).contiguous() # M x N, each row i: sim(shape_i, text_1), sim(shape_i, text_2),sim(shape_i, text_3) ...
        
        P_TS_distr = F.softmax(M, dim=1) # N x M 
        P_ST_distr = F.softmax(M_t, dim=1)  # M x N 
        # text-shape-text round trip 
        P_TST = torch.mm(P_TS_distr, P_ST_distr) # N x N 
        # build walk stastistics using equality matrix 
        estimate_error = self.build_walk_stastistics(P_TST, equality_matrix)
        
        ################################################
        # may be we should use mse instead of soft label cross entropy loss 
        ################################################
        # we will first take log function on P_TST and then applity softmax 
        L_TST_r = self.cross_entropy(P_TST, p_target)

        ################################################
        ## To associate text descriptions with all possible matching shapes
        ## we impose loss on the probability of asscociating each shape (j) with 
        ## any descriptions
        ################################################
        P_visit = torch.mean(P_TS_distr, dim=0, keepdim=True) # N text, M shape, (N x M) => 1 x M 
        
        soft_target2 = torch.ones(1, P_visit.size(1)).type_as(P_visit.data)/P_visit.size(1) # 1 x M
        L_TST_h = self.cross_entropy(P_visit, soft_target2)

        Total_loss = L_TST_r + self.lmbda * L_TST_h

        return Total_loss, P_TST, p_target


class LBA_Loss(nn.Module):
    def __init__(self, lmbda=1.0, LBA_model_type='MM', batch_size=None):
        super(LBA_Loss, self).__init__() 
        self.LBA_model_type = LBA_model_type 
        self.lmbda = lmbda

        if batch_size is not None:
            self.batch_size = batch_size 
        else:
            if self.LBA_model_type == 'MM' or self.LBA_model_type == 'STS':
                raise ValueError('please input the batch_size if LBA_model_type is MM or STS.')

        self.semisup_loss = Semisup_Loss(self.lmbda, lba_dist_type='standard')

    def forward(self, text_embedding, shape_embedding, labels): 
        """
        note that the returned P_STS and P_Target_TST e.t.c are nothing but for display purpose
        """
        # pdb.set_trace()
        # during test when we use this criterion, we may not get self.batch_size data 
        # so ..
        self.batch_size = text_embedding.size(0)

        if self.LBA_model_type == 'MM' or self.LBA_model_type == 'TST': 
            A = text_embedding
            B = shape_embedding 
            # TST_loss = L^{TST}_R + \lambda * L^{TST}_H, see equation (1) in the text2shape paper 
            TST_loss, P_TST, P_target_TST = self.semisup_loss(A, B, labels) 
        if self.LBA_model_type == 'MM' or self.LBA_model_type == 'STS':
            B = text_embedding
            A = shape_embedding
 
            labels = torch.from_numpy(np.array(range(self.batch_size))).type_as(A.data)
            # see equation (3) in the paper 
            # STS_loss = L^{TST}
            STS_loss, P_STS, P_target_STS = self.semisup_loss(A, B, labels)
    

        if self.LBA_model_type == 'MM':
            # see equaiton (3) in the text2shape paper 
        
            return TST_loss + STS_loss, P_TST, P_target_TST

        if self.LBA_model_type == 'TST':
            return TST_loss, P_TST, P_target_TST

        if self.LBA_model_type == 'STS': 
            return STS_loss, P_STS, P_target_STS


################################################################### 
## Classification loss 
###################################################################
class Classification_Loss(nn.Module):
    def __init__(self):
        super(Classification_Loss, self).__init__() 
        self.cross_entropy = nn.CrossEntropyLoss() # combines nn.LogSoftmax() and nn.NLLLoss()
    def forward(self, shape_output, shape_label_batch):
        loss = self.cross_entropy(shape_output, shape_label_batch)

        return loss 

################################################################### 
## Metric loss 
###################################################################
class Metric_Loss(nn.Module):
    """
    used only for training 
    """
    def __init__(self, opts, LBA_inverted_loss=True, LBA_normalized=True, LBA_max_norm=None):
        super(Metric_Loss, self).__init__() 
        # either is true 
        assert (LBA_inverted_loss is True) or (LBA_normalized is True)
        assert opts.LBA_n_captions_per_model == 2 
        self.LBA_inverted_loss = LBA_inverted_loss 
        self.LBA_normalized = LBA_normalized
        self.dataset = opts.dataset
        self.LBA_n_captions_per_model = opts.LBA_n_captions_per_model
        self.batch_size = opts.batch_size
        self.LBA_cosin_dist = opts.LBA_cosin_dist
        if self.dataset == 'primitives':
            self.LBA_n_primitive_shapes_per_category = opts.LBA_n_primitive_shapes_per_category
            assert self.LBA_n_primitive_shapes_per_category == 2 
        if LBA_inverted_loss is True: 
            self.cur_margin = 1. 
        else: 
            self.cur_margin = 0.5 

        ################################################
        ## should we specify the self.text_norm_weight and self.shape_norm_weight 
        ## here we add a penalty on the embedding norms  
        ################################################
        if LBA_max_norm is not None: 
            self.LBA_max_norm = LBA_max_norm
            self.text_norm_weight = 2.0 
            self.shape_norm_weight = 2.0 
        else: # default value 
            self.LBA_max_norm = LBA_max_norm
            self.text_norm_weight = 2.0 
            self.shape_norm_weight = 2.0 


    #######################################################
    ##
    #######################################################
    def cosine_similarity(self, X, Y):

        """ Copied from sklearn.metrics.pairwise.cosine_similarity
        Note that we assume X, Y has been normalized 

        Compute cosine similarity between samples in X and Y.
        Cosine similarity, or the cosine kernel, computes similarity as the
        normalized dot product of X and Y:
            K(X, Y) = <X, Y> / (||X||*||Y||)
        On L2-normalized data, this function is equivalent to linear_kernel.
        Read more in the :ref:`User Guide <cosine_similarity>`.
        Parameters
        ----------
        X : ndarray or sparse array, shape: (n_samples_X, n_features)
            Input data.
        Y : ndarray or sparse array, shape: (n_samples_Y, n_features)
            Input data. If ``None``, the output will be the pairwise
            similarities between all samples in ``X``.
        dense_output : boolean (optional), default True
            Whether to return dense output even when the input is sparse. If
            ``False``, the output is sparse if both input arrays are sparse.
            .. versionadded:: 0.17
               parameter ``dense_output`` for dense output.
        Returns
        -------
        kernel matrix : array
            An array with shape (n_samples_X, n_samples_Y).
        """
        Y_t = Y.transpose(0, 1)

        K = torch.mm(X, Y_t)

        return K

    def euclidean_distance(self, X, Y):

        """ Copied from sklearn.metrics.pairwise.cosine_similarity
        Note that we assume X, Y has been normalized 

        Compute cosine similarity between samples in X and Y.
        Cosine similarity, or the cosine kernel, computes similarity as the
        normalized dot product of X and Y:
            K(X, Y) = <X, Y> / (||X||*||Y||)
        On L2-normalized data, this function is equivalent to linear_kernel.
        Read more in the :ref:`User Guide <cosine_similarity>`.
        Parameters
        ----------
        X : ndarray or sparse array, shape: (n_samples_X, n_features)
            Input data.
        Y : ndarray or sparse array, shape: (n_samples_Y, n_features)
            Input data. If ``None``, the output will be the pairwise
            similarities between all samples in ``X``.
        dense_output : boolean (optional), default True
            Whether to return dense output even when the input is sparse. If
            ``False``, the output is sparse if both input arrays are sparse.
            .. versionadded:: 0.17
               parameter ``dense_output`` for dense output.
        Returns
        -------
        kernel matrix : array
            An array with shape (n_samples_X, n_samples_Y).
        """
        m, p = X.size() 
        n, p = Y.size() 
        X_exp = torch.stack([X]*n).transpose(0,1)
        Y_exp = torch.stack([Y]*m)
        dist = torch.sum((X_exp-Y_exp)**2,2).squeeze() # size: m x n 
        dist = (dist+1e-8).sqrt_() # applies inplace sqrt 

        return dist


    #######################################################
    ##
    ########################################################
    def smoothed_metric_loss(self, input_tensor, margin=1): 
        """
         Song et al., Deep Metric Learning via Lifted Structured Feature Embedding
        input_tensor: size: N x emb_size 
        """ 
        # compute pairwise distance 
        X = input_tensor # N x emb_size 
        m = margin 

        if self.LBA_cosin_dist is True: 
            assert (self.LBA_normalized is True) or (self.LBA_inverted_loss is True) 
            assert (self.LBA_normalized is True) and (margin < 1) or (self.LBA_inverted_loss is True)

            D = self.cosine_similarity(X, X)

            if self.LBA_inverted_loss is False: 
                D = 1.0 - D 
            else: 
                D /= 128. 
        else: 
            D = self.euclidean_distance(X, X)

        if self.LBA_inverted_loss is True:
            expmD = torch.exp(m + D)
        else: 
            expmD = torch.exp(m - D)

        # compute the loss 
        # assume that the input data is aligned in a way that two consective data form a pair 

        # L_{ij} = \log (\sum_{i, k} exp\{m - D_{ik}\} + \sum_{j, l} exp\{m - D_{jl}\}) + D_{ij}
        # L = \frac{1}{2|P|}\sum_{(i,j)\in P} \max(0, J_{i,j})^2
        J_all = [] 
        for pair_ind in range(self.batch_size//2): 
            i = pair_ind * 2 # 0, 2, 4, ...
            j = i + 1 # j is the postive of i 

            # the rest is the negative indices 
            # 0, ..., i-1, exclude(i, i+1),  i + 2, ..., self.batch_size
            ind_rest = np.hstack([np.arange(0, pair_ind * 2), np.arange(pair_ind * 2 + 2, self.batch_size)])
            neg_inds = [[i, k] for k in ind_rest]
            neg_inds.extend([[j, l] for l in ind_rest])

            neg_row_ids = [int(coord[0]) for coord in neg_inds]
            neg_col_ids = [int(coord[1]) for coord in neg_inds]
            neg_inds = [neg_row_ids, neg_col_ids]

            if self.LBA_inverted_loss is True: 
                J_ij = torch.log(torch.sum(expmD[neg_inds])) - D[i, j]
            else: 
                J_ij = torch.log(torch.sum(expmD[neg_inds])) + D[i, j]

            J_all.append(J_ij) 

        # convert list to tensor 
        # cannot convert t
        J_all = torch.stack(J_all)
        loss = torch.mean(F.relu(J_all)**2) * 0.5
    
        return loss 



    def forward(self, text_embeddings, shape_embeddings):
        
        # we may rewrite batch_size 
        self.batch_size = text_embeddings.size(0)
        if self.dataset == 'shapenet':
            # if self.batch_size = 2, indices = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7] 
            indices = [i // 2 for i in range(self.batch_size * self.LBA_n_captions_per_model)]
            shape_embeddings_rep = torch.index_select(shape_embeddings, 0, torch.LongTensor(indices))
        elif self.dataset == 'primitives': 
            shape_embeddings_rep = shape_embeddings 
        else: 
            raise ValueError('please select a valid dataset.')

        ##############################################################
        ## TT loss 
        ##############################################################
        embeddings = text_embeddings  
        metric_tt_loss= self.smoothed_metric_loss(embeddings, self.cur_margin) 

        ##############################################################
        ## ST loss (cross modality loss)
        ##############################################################
        if self.dataset == 'shapenet':
            mask_ndarray = np.asarray([0., 1.] * self.batch_size)[:, np.newaxis]
        elif self.dataset == 'primitives':
            assert self.LBA_n_primitive_shapes_per_category == 2
            assert self.batch_size % self.LBA_n_primitive_shapes_per_category == 0
            mask_ndarray = np.asarray([0., 1.] * (self.batch_size
                                      // self.LBA_n_primitive_shapes_per_category))[:, np.newaxis]
        else:
            raise ValueError('Please select a valid datset.')
        # Tensor, with value (when batch size is 10)
        #  0 1 0 1 0 1 0 1 0 1 => expand: bz x 128
        mask = torch.from_numpy(mask_ndarray).float().type_as(text_embeddings.data).expand(self.batch_size, text_embeddings.size(1))
        inverted_mask = 1. - mask
        # embeddings is : 
        # text_1_emb, shape_emb_1, ..., text_N_emb, shape_emb_N (the consective two are the same label)
        embeddings = text_embeddings * mask + shape_embeddings_rep * inverted_mask
        metric_st_loss = self.smoothed_metric_loss(embeddings,self.cur_margin)

        # embeddings = text_embeddings * inverted_mask + shape_embeddings_rep * mask
        # metric_ts_loss = self.smoothed_metric_loss(embeddings, name='smoothed_metric_loss_ts', margin=cur_margin)

        Total_loss = metric_tt_loss + 2. * metric_st_loss


        if self.LBA_normalized is False:  # Add a penalty on the embedding norms
            """
            only when self.LBA_normalizd is False
            """
            text_norms = torch.norm(text_embeddings, p=2, dim=1)
            unweighted_txt_loss = torch.mean(F.relu(text_norms - self.LBA_max_norm))
            shape_norms = torch.norm(shape_embeddings, p=2, dim=1)
            unweighted_shape_loss = torch.mean(F.relu(shape_norms - self.LBA_max_norm))

            Total_loss_with_norm = Total_loss + self.text_norm_weight * unweighted_txt_loss + self.shape_norm_weight * unweighted_shape_loss
            
            return Total_loss_with_norm
        else: 
            return Total_loss 

 
