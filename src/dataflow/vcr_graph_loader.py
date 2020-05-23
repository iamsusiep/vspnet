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
        img_ids_rcnn = pickle.loads(self.txn.get("keys".encode('utf-8')))

        fn1, fn2 = '/home/suji/spring20/vspnet/src/dataflow/srl_entries_trip.pkl', '/home/suji/spring20/vspnet/src/dataflow/vcr_embed.pkl'
        
        with open(fn1, 'rb') as handle:
            self.srl_entries = pickle.load(handle)        
        with open(fn2, 'rb') as handle:
            self.vcr_embed= pickle.load(handle)      
        self.img_ids = np.asarray(list(set(self.srl_entries.keys()) &set(img_ids_rcnn)))
        self.imgid2idx = {int(imgid.split('-')[1]): idx for idx, imgid in enumerate(self.img_ids)}
        self.size = self.img_ids.size
        self.emb_dim = 300
        self.verb_emb = self.vcr_embed['verb_emd']
        self.noun_emb = self.vcr_embed['noun_emd']

        self.verb2idx = {}
        self.noun2idx = {}
        for i, v in enumerate(list(self.verb_emb.keys())):
            self.verb2idx[v] = i + 1
        for i, v in enumerate(list(self.noun_emb.keys())):
            self.noun2idx[v] = i + 1
        with open("/home/suji/spring20/vspnet/src/dataflow/dictionary.pkl", 'rb') as handle:
            self.dictionary= pickle.load(handle)  

    def get_gt_batch(self, idx_list, pack):
        img_ids = [self.img_ids[idx] for idx in idx_list]
        features = []
        prop_box = []

        ent_embs =[]
        pred_embs = []
        ent_lbls = []
        pred_lbls = []
        pred_roles = []
        ent_boxes= []
        for img_id in img_ids:
            pred_emb = []
            pred_lbl = []
            ent_emb = []
            ent_lbl = []
            triplets = []
            annot_dict = self.srl_entries[img_id]
            for annot_id, srl_list in annot_dict.items():
                for srl in srl_list:
                    sbj = []
                    pred = None
                    obj = []
                    if 'ARG0' in srl.keys() and 'ARG1'in srl.keys():
                        for n in srl['ARG0']:
                            if n in self.noun_emb:
                                sbj.append(n)
                        for n in srl['ARG1']:
                            if n in self.noun_emb:
                                obj.append(n)
                        pred= srl['V']
                    elif 'ARG1' in srl.keys() and 'ARG2'in srl.keys():
                        for n in srl['ARG1']:
                            if n in self.noun_emb:
                                sbj.append(n)
                        for n in srl['ARG2']:
                            if n in self.noun_emb:
                                obj.append(n)
                        pred= srl['V']
                    else: # one argument,,, sbj
                        if 'ARG0' in srl.keys():
                            for n in srl['ARG0']:
                                if n in self.noun_emb:
                                    sbj.append(n)
                        elif 'ARG1' in srl.keys():
                            for n in srl['ARG1']:
                                if n in self.noun_emb:
                                    sbj.append(n)
                        elif 'ARG2' in srl.keys():
                            for n in srl['ARG2']:
                                if n in self.noun_emb:
                                    sbj.append(n)
                        pred= srl['V']
                    new_sbj = []
                    new_obj = []
                    for n in sbj:
                        if n in self.dictionary.keys():
                            new_sbj.append(n)
                    for n in obj:
                        if n in self.dictionary.keys():
                            new_obj.append(n)
                    for s in new_sbj:
                        for o in new_obj:
                            #print(s, o)
                            triplets.append((s, pred,o))

            pred_sbj_rc = []
            pred_obj_rc = []
        
            pred_emb, pred_lbl, ent_emb, ent_lbl= [], [],[], []
            
            for sbj, pred, obj in triplets:
                ent_list, pred_list, ent_emb_list, pred_emb_list = [], [],[], []
                pred_emb.append(self.verb_emb[pred])
                pred_lbl.append(self.verb2idx[pred])
                ent_emb.append(self.noun_emb[sbj])
                ent_lbl.append(self.noun2idx[sbj])
                pred_sbj_rc.append((len(pred_lbl)-1, len(ent_lbl)-1))
                ent_emb.append(self.noun_emb[obj])
                ent_lbl.append(self.noun2idx[obj])
                pred_obj_rc.append((len(pred_lbl)-1, len(ent_lbl)-1))
                '''
                for n in sbj:
                    ent_emb.append(self.noun_emb[n])
                    ent_lbl.append(self.noun2idx[n])
                    pred_sbj_rc.append((len(pred_lbl)-1, len(ent_lbl)-1))
                for n in obj:
                    ent_emb.append(self.noun_emb[n])
                    ent_lbl.append(self.noun2idx[n])
                    pred_obj_rc.append((len(pred_lbl)-1, len(ent_lbl)-1))
                '''
            pred_embs.append(np.asarray(pred_emb))
            pred_lbls.append(np.asarray(pred_lbl))
            ent_embs.append(np.asarray(ent_emb))
            ent_lbls.append(np.asarray(ent_lbl))
            ent_boxes.append(np.zeros((len(ent_lbl), 4)))
            pred_sbj_roles = np.zeros((len(pred_lbl), len(ent_lbl)), dtype=bool)
            pred_obj_roles = np.zeros((len(pred_lbl), len(ent_lbl)), dtype=bool)
            for r,c in pred_sbj_rc:
                pred_sbj_roles[r][c] = 1
            for r,c in pred_obj_rc:
                pred_obj_roles[r][c] = 1

            #print('pred_sbj_roles', pred_sbj_roles)
            #print("pred_obj_roles", pred_obj_roles)
            pred_roles.append(np.stack((pred_sbj_roles, pred_obj_roles)))
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
        gt_graph['ent_emb'] = np.asarray(ent_embs)
        gt_graph['pred_emb'] = np.asarray(pred_embs)
        gt_graph['ent_lbl'] = np.asarray(ent_lbls)
        gt_graph['pred_lbl'] = np.asarray(pred_lbls)
        gt_graph['num_ent'] = np.asarray([len(x) for x in ent_embs])
        gt_graph['num_pred'] = np.asarray([len(x) for x in pred_embs])
        gt_graph['pred_roles']=pred_roles
        gt_graph['ent_box']  = ent_boxes#np.zeros((ent_lbls.shape[0], 4))
        #gt_graph['pred_roles'] =[np.zeros((2, 1,1)) for idx in idx_list]
        # print(gt_graph['pred_lbl'].shape)
        # print(gt_graph['ent_lbl'].shape)
        #print(np.zeros((2,  gt_graph['pred_lbl'].shape[1],gt_graph['ent_lbl'].shape[1])))
        '''
        if len(gt_graph['pred_lbl'].shape) == 2 and len(gt_graph['ent_lbl'].shape) == 2:
            gt_graph['pred_roles'] = [np.zeros((2,  gt_graph['pred_lbl'].shape[1],gt_graph['ent_lbl'].shape[1])) for idx in idx_list]
        else:
            gt_graph['pred_roles'] = [np.zeros((2, 1,1)) for idx in idx_list]
        '''
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
