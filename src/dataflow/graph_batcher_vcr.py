import numpy as np
class GraphBatcher:

    def __init__(self, loader, batch_size):
        self.loader = loader
        self.batch_size = batch_size

        self.cursor = 0
        self.subset_idx = np.arange(loader.size)
        self.size = loader.size
        self.num_batch = int(np.ceil(self.subset_idx.shape[0] / self.batch_size))

    def set_subset(self, is_train=True):
        if is_train:
            self.subset_idx = np.asarray(self.subset_idx[:int(self.size*0.8)], dtype='int32').flatten()
            self.size = self.subset_idx.shape[0]
        else:
            self.subset_idx = np.asarray(self.subset_idx[int(self.size*0.8):], dtype='int32').flatten()
            self.size = self.subset_idx.shape[0]
        #self.subset_idx = np.asarray(idx, dtype='int32').flatten()
        #self.size = self.subset_idx.shape[0]
        self.num_batch = int(np.ceil(self.subset_idx.shape[0] / self.batch_size))
        

    def reset(self):
        self.cursor = 0

    def next_batch(self, keep_cursor=False):
        new_cursor = min(self.cursor + self.batch_size, self.subset_idx.shape[0])
        idx_idx = np.arange(self.cursor, new_cursor)
        idx = self.subset_idx[idx_idx]

        gt_graph = self.loader.get_gt_batch(idx, pack=True)          
        feed_dict = dict(gt_graph)

        
        if not keep_cursor:
            self.cursor = new_cursor
            if self.cursor >= self.size:
                self.cursor = 0        
        
        return feed_dict
