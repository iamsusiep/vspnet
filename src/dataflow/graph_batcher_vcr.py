import os
import numpy as np
import lmdb
import pickle
class GraphBatcher:

    def __init__(self, batch_size, dim_feats=4096,num_proposals=20):
        self.batch_size = batch_size

        self.cursor = 0

        lmdb_path = '/home/suji/spring20/vspnet/src/dataflow/detection_test'
        self.env = lmdb.open(lmdb_path, map_size=1e12, readonly=True, lock=False)
        self.txn = self.env.begin(write=False)
        self.num_proposals = num_proposals
        self.dim_feats = dim_feats
        self.img_ids = np.array(pickle.loads(self.txn.get("keys".encode('utf-8'))))

        self.imgid2idx = {imgid: idx for idx, imgid in enumerate(self.img_ids)}
        self.subset_idx = np.arange(len(self.img_ids))
        self.size = len(self.img_ids)
        self.num_batch = int(np.ceil(self.subset_idx.shape[0] / self.batch_size))

        fn1, fn2 = '/home/suji/spring20/vspnet/src/dataflow/srl_entries.pkl', '/home/suji/spring20/vspnet/src/dataflow/vcr_embed.pkl'
        
        with open(fn1, 'rb') as handle:
            self.srl_entries = pickle.load(handle)        
        with open(fn2, 'rb') as handle:
            self.vcr_embed= pickle.load(handle)       
        self.emb_dim = 300
        self.verb_emb = self.vcr_embed['verb_emd']
        self.noun_emb = self.vcr_embed['noun_emd']

        self.verb2idx = {}
        self.noun2idx = {}
        for i, v in enumerate(list(self.verb_emb.keys())):
            self.verb2idx[v] = i + 1
        for i, v in enumerate(list(self.noun_emb.keys())):
            self.noun2idx[v] = i + 1

    def set_subset(self, idx):
        self.subset_idx = np.asarray(idx, dtype='int32').flatten()
        self.size = self.subset_idx.shape[0]
        self.num_batch = int(np.ceil(self.subset_idx.shape[0] / self.batch_size))
        

    def reset(self):
        self.cursor = 0

    def next_batch(self, keep_cursor=False):
        new_cursor = min(self.cursor + self.batch_size, self.subset_idx.shape[0])
        idx_idx = np.arange(self.cursor, new_cursor)
        idx = self.subset_idx[idx_idx]
        feed_dict = dict()

        img_ids = self.img_ids[idx]
        features = []
        prop_box = []

        ent_embs =[]
        pred_embs = []
        ent_lbls = []
        pred_lbls = []
        for img_id in img_ids:
            pred_emb = []
            pred_lbl = []
            ent_emb = []
            ent_lbl = []
            if img_id in self.srl_entries:
                annot_dict = self.srl_entries[img_id]
                for annot_id, srl_list in annot_dict.items():
                    for srl in srl_list:
                        for k, v in srl.items():
                            if k == 'V':
                                pred_emb.append(self.verb_emb[v])
                                pred_lbl.append(self.verb2idx[v])
                            else:
                                for n in v:
                                    if n in self.noun_emb:
                                        ent_emb.append(self.noun_emb[n])
                                        ent_lbl.append(self.noun2idx[n])
            else:
                pred_emb.append(np.zeros(300))
                pred_lbl.append(np.zeros(1))
                ent_emb.append(np.zeros(300))
                ent_lbl.append(np.zeros(1))
            pred_embs.append(pred_emb)
            pred_lbls.append(pred_lbl)
            ent_embs.append(ent_emb)
            ent_lbls.append(ent_lbl)
            item = pickle.loads(self.txn.get(str(img_id).encode('utf-8')))
            ft = item['feat']
            ft = np.frombuffer(ft, 'float32')
            ft = np.reshape(ft, (-1, self.dim_feats))

            height, width = item['img_info']['height'], item['img_info']['width'] 
            height_bbox, width_bbox = item['bbox_info']['height'], item['bbox_info']['width'] 
            scale = width/width_bbox
            
            pb = item['bbox']
            pb =[[y * scale for y in x ] for x in pb]
            pb = np.asarray([bbox for bbox, score in sorted(zip(pb, item['pred_scores']), key=lambda t: t[1], reverse =True)])

            if self.num_proposals != None: 
                if ft.shape[0] > self.num_proposals:
                    ft = ft[:self.num_proposals]
                elif ft.shape[0] < self.num_proposals:
                    ft = np.concatenate((ft, np.zeros((self.num_proposals - ft.shape[0], self.dim_feats), dtype=ft.dtype)), axis=0)
                if pb.shape[0] > self.num_proposals:
                    pb = pb[:self.num_proposals]
                elif pb.shape[0] < self.num_proposals:
                    pb = np.concatenate((pb, np.zeros((self.num_proposals - pb.shape[0], 4), dtype=pb.dtype)), axis=0)
            features.append(ft)
            prop_box.append(pb)

        features = np.stack(features)  
        prop_box = np.asarray(prop_box)
        ent_embs = np.asarray(ent_embs)
        pred_embs = np.asarray(pred_embs)

        feed_dict['image_id'] = img_ids
        feed_dict['proposal_features'] = features
        feed_dict['proposal_boxes'] = prop_box
        feed_dict['ent_emb'] = ent_embs
        feed_dict['pred_emb'] = pred_embs
        feed_dict['ent_lbl'] = ent_lbls 
        feed_dict['pred_lbl'] = pred_lbls
        feed_dict['num_ent'] = [len(x) for x in ent_embs]
        feed_dict['num_pred'] = [len(x) for x in pred_embs]

        
        if not keep_cursor:
            self.cursor = new_cursor
            if self.cursor >= self.size:
                self.cursor = 0        
        
        return feed_dict
