import os
import lmdb
import numpy as np
# from .util import load_preproc_image
import pickle

class GraphBatcher:
    '''
                'image_id': image_id,
                'ent_lbl': ent_lbl,
                'ent_box': ent_box,
                'pred_lbl': pred_lbl,
                'pred_roles': pred_roles,
                'ent_emb': ent_emb,
                'pred_emb': pred_emb,
                'num_ent': num_entities,
                'num_pred': num_preds,
            }
    '''
    def __init__(self, batch_size):
        self.batch_size = batch_size
        
        self.cursor = 0
        self.subset_idx = np.arange(loader.size)
        self.size = loader.size
        # self.num_batch = int(np.ceil(self.subset_idx.shape[0] / self.batch_size))
        
        # self.seed = seed
        # self.rand_gen = np.random.RandomState(seed)
        # self.rand_samp = self.rand_gen.choice
        
        # self.env = lmdb.open(lmdb_path, map_size=1e12, readonly=True, lock=False)
        # self.txn = self.env.begin(write=False)
        
        self.dim_feats = dim_feats 

        # input_file = ''
        with open(r"/home/suji/spring20/stuffs.pkl", "rb") as input_file:
            self.d = cPickle.load(input_file) # dict_keys(['feat_tensor', 'feat_np', 'proposal', 'img_id', 'orig_bb', 'bb'])

        
    def set_subset(self, idx):
        self.subset_idx = np.asarray(idx, dtype='int32').flatten()
        self.size = self.subset_idx.shape[0]
        self.num_batch = int(np.ceil(self.subset_idx.shape[0] / self.batch_size))
        
    def reset(self):
        self.cursor = 0
        
        
    def next_batch(self, keep_cursor=False):
        idx_idx = np.arange(self.cursor, new_cursor)
        idx = self.subset_idx[idx_idx]
        new_cursor = min(self.cursor + self.batch_size, self.subset_idx.shape[0])
        feed_dict = dict()
        feed_dict['image_id'] = np.array(self.d['img_id'])
        feed_dict['proposal_features'] = np.array(self.d['features'])
        feed_dict['proposal_boxes'] = np.array(self.d['scaled_bblist']) #x1, y1, x2, y2

        # if not keep_cursor:
        #     self.cursor = new_cursor
        #     if self.cursor >= self.size:
        #         self.cursor = 0        
        
        # feed_dict['ent_lbl'] =None
        # feed_dict['ent_box'] = None
        # feed_dict['pred_lbl'] = None
        # feed_dict['pred_roles'] = None
        # feed_dict['ent_emb'] = None
        # feed_dict['pred_emb'] = None
        # feed_dict['num_ent'] = None
        # feed_dict['num_pred'] = None

        return feed_dict
'''