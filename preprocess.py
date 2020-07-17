import time
import json
import numpy as np

from itertools import groupby
from sklearn.utils import shuffle
from scipy.sparse import *

class dataprocessing:
    def __init__(self, path):
        self.t1 = time.time()
        f = open(path+'/real_train.json', encoding='utf-8')
        tr = f.read()
        f.close()
        self.train = json.loads(tr)

        f = open(path+'/real_val.json', encoding='utf-8')
        vl = f.read()
        f.close()
        self.val = json.loads(vl)

    def get_list(self):
        total = self.train + self.val
        self.total_songs = [q['songs'] for q in total]
        self.tr_songs = self.total_songs[:len(self.train)]
        self.te_songs = self.total_songs[len(self.train):]
        self.tr_tags = [q['tags'] for q in self.train]
        self.te_tags = [q['tags'] for q in self.val]
        self.tr_ids = [q['id'] for q in self.train]
        self.te_ids = [q['id'] for q in self.val]

    def get_mtx(self):
        train = []
        sid_to_idx = {}
        tag_to_idx = {}
        idx = 0

        for i, songs in enumerate(self.tr_songs):
            view = songs
            for song_id in view:
                if song_id not in sid_to_idx:
                    sid_to_idx[song_id] = idx
                    idx+=1
            view = [sid_to_idx[song] for song in view]
            train.append(view)
        
        idx = 0
        n_items = len(sid_to_idx)
        self.n_items = n_items

        for i, tags in enumerate(self.tr_tags):
            for tag in tags:
                if tag not in tag_to_idx: #Extend song length backward
                    tag_to_idx[tag] = n_items + idx
                    idx+=1
            train[i].extend([tag_to_idx[tag] for tag in tags])
        
        n_tags = len(tag_to_idx)
        self.n_tags = n_tags

        test = []
        idx = 0
        for i, songs in enumerate(self.te_songs):
            view = songs
            temp = []
            for song_id in view:
                if song_id not in sid_to_idx:
                    continue # test songs는 dict에 넣지 않음
                temp.append(sid_to_idx[song_id])
            test.append(temp)    

        idx = 0
        for i, tags in enumerate(self.te_tags):
            temp = []
            for tag in tags:
                if tag not in tag_to_idx:
                    continue # test tags는 dict에 넣지 않음
                temp.append(tag)
            test[i].extend([tag_to_idx[tag] for tag in temp])
        
        train = shuffle(train)
        self.idx_to_sid = {x:y for (y,x) in sid_to_idx.items()}
        self.idx_to_tag = {(x-n_items):y for (y,x) in tag_to_idx.items()}

        train_mtx = lil_matrix((len(train), n_items+n_tags))
        test_mtx = lil_matrix((len(test), n_items+n_tags))

        for p_id, playlist in enumerate(train):
            for item in playlist: #itme = song or tag
                train_mtx[p_id, item] = 1

        for p_id, playlist in enumerate(test):
            for item in playlist:
                test_mtx[p_id, item] = 1
        
        self.train_mtx = csr_matrix(train_mtx)
        #self.test_mtx = csr_matrix(test_mtx)
        #일단 only train use
        print("rating matrix shape is:", np.shape(self.train_mtx))
        print("Total time to preprocessing:",round(time.time()-self.t1,3))
    def run(self):
        self.get_list()
        self.get_mtx()