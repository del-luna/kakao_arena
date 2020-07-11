import time
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.utils import shuffle
from scipy.sparse import lil_matrix, csr_matrix, vstack
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

random.seed(777)

class Melon_generator:
    def __init__(self, train_path, val_path):
        
        f = open(train_path, encoding='utf-8')
        js = f.read()
        f.close()
        self.train = json.loads(js)
        
        f = open(val_path, encoding='utf-8')
        vl = f.read()
        f.close()
        self.val = json.loads(vl)

        
        self.no_seed = {idx:play['plylst_title'] for idx,play in enumerate(self.val) if len(play['songs'])==0}
        self.no_seed_idx = list(self.no_seed.keys())
        self.no_seed_titles = list(self.no_seed.values())
        self.train_titles = [play['plylst_title'] for play in self.train]
        self.random_titles = random.sample(self.train_titles, 40000)
        
        self.title_info = self.random_titles + self.no_seed_titles
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(self.title_info)
        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        
    def get_rec_name(self, title, idx, cosine_sim):
        sim_scores = list(enumerate(cosine_sim[len(self.random_titles)+idx]))
        sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11] #본인제외
        self.move_indices =  [i[0] for i in sim_scores]
        return title, [self.title_info[i] for i in self.move_indices]
    
    def add_seed(self):
        self.seed = []
        self.error_idx = []
        df_train = pd.read_json('../data/real_train.json',encoding ='utf-8')
        for idx in tqdm(range(len(self.no_seed_idx))):
            _, title = self.get_rec_name(self.no_seed_titles[idx], idx, self.cosine_sim)
            for t in title:
                if t in self.train_titles:
                    self.seed.append(t)
                    break
                elif t == title[-1] and t not in self.train_titles:
                    print(idx,"error! no seed!")
                    self.error_idx.append(idx)
        self.seed_tracks = []
        for title in tqdm(self.seed):
            self.seed_tracks.append(df_train[df_train['plylst_title'] == title].songs.tolist()[0])
        for idx, tracks in zip(self.no_seed_idx, self.seed_tracks):
            self.val[idx]['songs']=tracks
            
    def get_list(self):
        total = self.train + self.val
        self.total_songs = [q['songs'] for q in total]
        self.total_like = [q['like_cnt'] for q in total]
        self.tr_songs = self.total_songs[:len(self.train)]
        self.te_songs = self.total_songs[len(self.train):]
        self.tr_tags = [q['tags'] for q in self.train]
        self.te_tags = [q['tags'] for q in self.val]
        self.te_ids = [q['id'] for q in self.val]
    
    def normalize_like_cnt(self):
        self.song_to_like = {}
        for idx, view in enumerate(self.total_songs):
            for song in view:
                if song not in self.song_to_like:
                    self.song_to_like[song] = self.total_like[idx]
                if song in self.song_to_like:
                    self.song_to_like[song] = int((self.song_to_like[song] + self.total_like[idx])/2)
        like_value = list(self.song_to_like.values())
        like_key = list(self.song_to_like.keys())
        max_like = max(like_value)
        min_like = min(like_value)
        normal_value = [round((value-min_like)/(max_like-min_like)*4 + 1,3) for value in like_value]
        self.song_like = [(k,v) for k,v in zip(like_key, normal_value)]
           
    def processing(self):
        t1 = time.time()
        tr = []
        sid_to_idx = {}
        tag_to_idx = {}
        idx = 0
        
        for i, songs in enumerate(self.tr_songs):
            view = songs
            for item_id in view:
                if item_id not in sid_to_idx:
                    sid_to_idx[item_id] = idx # {song_id : idx}
                    idx+=1
            view = [sid_to_idx[song] for song in view]
            tr.append(view)
        
        idx = 0
        n_items = len(sid_to_idx)
        self.n_items=n_items
        
        for i, tags in enumerate(self.tr_tags):
            for tag in tags:
                if tag not in tag_to_idx:
                    #Extend song length backward
                    tag_to_idx[tag] = n_items + idx #{tag : idx}
                    idx += 1
            tr[i].extend([tag_to_idx[tag] for tag in tags]) # {playlist : song +tag}
        n_tags = len(tag_to_idx)
        self.n_tags = n_tags
        
        te = []
        temp = []
        idx = 0
        for i, songs in enumerate(self.te_songs):
            view = songs
            ret = []
            for item_id in view:
                if item_id not in sid_to_idx:
                    continue
                ret.append(sid_to_idx[item_id])
            te.append(ret)
        idx = 0
        for i, tags in enumerate(self.te_tags):
            ret = []
            for tag in tags:
                if tag not in tag_to_idx:
                    continue
                ret.append(tag)
            te[i].extend([tag_to_idx[tag] for tag in ret])
            
        tr = shuffle(tr)
        self.idx_to_sid = {x:y for (y,x) in sid_to_idx.items()}
        self.idx_to_tag = {(x-n_items):y for (y,x) in tag_to_idx.items()}
        
        tr_mtx = lil_matrix((len(tr), n_items+n_tags))
        te_mtx = lil_matrix((len(te), n_items+n_tags))
        
        for i, view in enumerate(tr):
            for idx, data in enumerate(view):
                if view[idx] > n_items-1: # if view[idx] == tag
                    tr_mtx[i, data] = 1
                    continue
                tr_mtx[i, data] = self.song_like[view[idx]][1] #rating == like
                
        for i, view in enumerate(te):
            for idx, data in enumerate(view):
                if view[idx] > n_items-1:
                    te_mtx[i, data] = 1
                    continue
                te_mtx[i, data] = self.song_like[view[idx]][1]
        self.tr_mtx = csr_matrix(tr_mtx)
        self.te_mtx = csr_matrix(te_mtx)
        r = vstack([te_mtx, tr_mtx])
        self.r = csr_matrix(r)
        print("rating matrix shape is :",np.shape(r))
        print("Time taken for Preprocessing:",round(time.time()-t1),3)
        
    def run(self):
        self.add_seed()
        self.get_list()
        self.normalize_like_cnt()
        self.processing()