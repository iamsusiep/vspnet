# from vg_graph_loader_1.py: loading stanford VG metadata instead of original

import os
import json
import pickle
import cv2
import numpy as np
import lmdb
# import tensorflow as tf

class VCRGraphLoader:
    def __init__(self, num_proposals=20, dim_feats=4096):
        lmdb_path = '/home/suji/spring20/vspnet/src/dataflow/detection_test'
        self.env = lmdb.open(lmdb_path, map_size=1e12, readonly=True, lock=False)
        self.txn = self.env.begin(write=False)
        self.num_proposals = num_proposals
        self.dim_feats = dim_feats
        self.img_ids = np.array(pickle.loads(self.txn.get("keys".encode('utf-8'))))
        self.size = self.img_ids.size

        # self.imgid2idx = {imgid: idx for idx, imgid in enumerate(self.img_ids)}

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


    def get_gt_batch(self, idx_list, pack):
        img_ids = [self.img_ids[idx] for idx in idx_list]
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
        gt_graph = dict()
        gt_graph['image_id'] = [imgid.split('-')[1] for imgid in img_ids]
        gt_graph['proposal_features'] = features
        gt_graph['proposal_boxes'] = prop_box
        gt_graph['ent_emb'] = ent_embs
        gt_graph['pred_emb'] = pred_embs
        gt_graph['ent_lbl'] = ent_lbls 
        gt_graph['pred_lbl'] = pred_lbls
        gt_graph['num_ent'] = np.asarray([len(x) for x in ent_embs])
        gt_graph['num_pred'] = np.asarray([len(x) for x in pred_embs])
        gt_graph['pred_roles'] =[np.zeros((2, 1,1)) for idx in idx_list]
        if pack:
            num_entities = []
            num_preds = []
            for i in range(len(idx_list)):
                num_entities.append(np.asarray(gt_graph['ent_emb'][i]).shape[0])
                num_preds.append(np.asarray(gt_graph['pred_emb'][i]).shape[0])
            num_entities = np.asarray(num_entities, dtype='int32')
            num_preds = np.asarray(num_preds, dtype='int32') 
            max_n_ent = np.max(num_entities)
            max_n_pred = np.max(num_preds)

            ent_lbl = np.zeros((len(idx_list), max_n_ent,), dtype='int32')
            pred_lbl = np.zeros((len(idx_list), max_n_pred,), dtype='int32')
            ent_emb = np.zeros((len(idx_list), max_n_ent, self.emb_dim), dtype='float32')
            pred_emb = np.zeros((len(idx_list), max_n_pred, self.emb_dim), dtype='float32')
            ent_box = np.zeros((len(idx_list), max_n_ent, 4), dtype='float32')
            pred_roles = np.zeros((len(idx_list), gt_graph['pred_roles'][0].shape[0], max_n_pred, max_n_ent), dtype='bool')

            for i in range(len(idx_list)):
                if max_n_ent > 0:
                    if num_entities[i] > 0:
                        ent_lbl[i, :num_entities[i]] = gt_graph['ent_lbl'][i]
                        ent_emb[i, :num_entities[i]] = gt_graph['ent_emb'][i]
                        #ent_box[i, :num_entities[i]] = gt_graph['ent_box'][i]
                if max_n_pred > 0:
                    assert(max_n_ent > 0)
                    if num_preds[i] > 0:
                        pred_lbl[i, :num_preds[i]] = gt_graph['pred_lbl'][i]
                        pred_emb[i, :num_preds[i]] = gt_graph['pred_emb'][i]
                        #if num_entities[i] > 0:
                        #    pred_roles[i, :, :num_preds[i], :num_entities[i]] = gt_graph['pred_roles'][i]

            image_id = np.asarray(gt_graph['image_id'])
                
            return {
                'image_id': image_id,
                'ent_lbl': ent_lbl,
                'ent_box': ent_box,
                'pred_lbl': pred_lbl,
                'pred_roles': pred_roles,
                'ent_emb': ent_emb,
                'pred_emb': pred_emb,
                'num_ent': num_entities,
                'num_pred': num_preds,
                'proposal_boxes': prop_box, 
                'proposal_features':features
            }
        
        return gt_graph
