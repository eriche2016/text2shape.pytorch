"""
Parallel data loading functions
"""
import sys 
import time 
import numpy as np 
from six.moves import queue 
from multiprocessing import Process, Event 


class DataProcess(Process): 
    def __init__(self, data_queue, data_paths, batch_size=None, repeat=True):
        """
        Args:
            data_queue: Multiprocessing queue 
            data_paths: List of (data, label) pairs used to load data
            batch_size: Batch size
            repeat: if set True, return data until exit is set.  
        """
        super(DataProcess, self).__init__() 
        if batch_size is None: 
            print('batch size is None, so we set it to 100 by default.')
            batch_size = 100 # by default, we set batch size to be 100 

        # Queue to transfer the loaded mini batches 
        self.data_queue = data_queue
        self.data_paths = data_paths 
        self.num_data = len(data_paths) 
        self.repeat = repeat 

        # Tuple of data shape 
        self.batch_size = batch_size
        self.exit = Event() 
        # shuffle database indices initially 
        self.shuffle_db_inds() 

        # Only for external use and is only approximate-don't rely on this when testing
        # when testing, set repeate to False and rely on the data process to quit!!!
        self.iters_per_epoch = self.num_data // self.batch_size

    def shuffle_db_inds(self): 
        # Randomly permute the training roidb 
        if self.repeat: # if repeat is set to be True 
            self.perm = np.random.permutation(np.arange(self.num_data)) 
        else:
            self.perm = np.arange(self.num_data)

        # every time we shuffle the data, that means we are in the start of an epoch 
        # thus we set self.cur to be 0 
        self.cur = 0 

    # this data process will return a minibatch everytime we call get_next_minibatch
    def get_next_minibatch(self): 
        if (self.cur + self.batch_size) >= self.num_data and self.repeat: 
            # we exceed the dataset and self.repeat is True
            # then we shuffle the indexs(which will set self.cur to be 0) 
            self.shuffle_db_inds()
        # take out data 
        db_inds = self.perm[self.cur:min(self.cur+self.batch_size, self.num_data)]
        
        # update self.cur 
        self.cur += self.batch_size
        return db_inds # just return index of examples in the minibatch 

    # this will shut down this data process 
    def shutdown(self): 
        self.exit.set()  

    # 要执行的任务代码写在run方法中
    # run method in Process:  Method representing the process’s activity, 
    #   You may override this method in a subclass.
    # we will call start() on the process (Start the process’s activity)
    # start() arranges for the object’s run() method to be invoked in a separate process. 
    def run(self): 
        # Run the loop until exit flag is set 
        while not self.exit.is_set() and self.cur < self.num_data: 
            # ensure that the network sees (almost) all data per epoch 
            db_inds = self.get_next_minibatch() 

            data_list = [] 
            label_list = [] 
            # start loading data and store it in data_list and label_list
            for db_ind in db_inds:  
                datum = self.load_datum(self.data_paths[db_ind]) 
                label = self.load_label(self.data_paths[db_ind]) 
                
                data_list.append(datum)
                label_list.append(label) 

            batch_data = np.array(data_list).astype(np.float32)
            batch_label = np.array(label_list).astype(np.float32)

            # The following will wait until the queue frees 
            # put tuple of minibatch data into the data queue 
            self.data_queue.put((batch_data, batch_label), block=True)

        def load_datum(self,path):
            pass 

        def load_label(self, path):
            pass 

# kill processes
def kill_processes(queue, processes): 
    print('signal processes to shutdown')

    for p in processes: 
        p.shutdown() 

    print("empty queue")
    #################################################
    ## The get method will be successful only when
    ## the item stored  in the queue are not tensor, but numpy array
    ## otherwise, we cannot run queue.get(False):
    ## ref: https://discuss.pytorch.org/t/using-torch-tensor-over-multiprocessing-queue-process-fails/2847/2 
    #################################################
    while not queue.empty(): # If queue is not empty 
        time.sleep(0.5)
        try: 
            queue.get(False) 
        except:
            print('now queue size is {0}'.format(queue.qsize()))
            pass 

    print('killing processes.') 
    for p in processes:
        p.terminate() 


def make_data_processes(data_process_class, queue, data_paths, opts, repeat): 
    """
    Make a set of data processes for parallel data loading 
    """
    processes = [] 
    for i in range(opts.num_workers): 
        process = data_process_class(queue, data_paths, opts, repeat=repeat)
        process.start() 
        processes.append(process) 

    return processes 

# During testing phase, we may need to use this method 
def get_while_running(data_process, data_queue, sleep_time=0): 
    while True:
        time.sleep(sleep_time)
        try:
            # quivalent to get(False) (Remove and return an item from the queue)
            batch_data = data_queue.get_nowait() 
        except queue.Empty:
            if not data_process.is_alive():
                break 
            else: 
                continue 

        yield batch_data 

if __name__ == '__main__':
    pass 












