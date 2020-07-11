import numpy as np
from scipy.sparse import *

from implicit.evaluation import  *
from implicit.als import AlternatingLeastSquares as ALS

def als_run(data, n_items, train, test, idx_to_sid, idx_to_tag):
    als_model = ALS(factors=128, regularization=0.08)
    als_model.fit(data.T * 15.0) #item-user matrix -> transpose
    item_model = ALS(use_gpu=False)
    tag_model = ALS(use_gpu=False)
    item_model.user_factors = als_model.user_factors
    tag_model.user_factors = als_model.user_factors
    item_model.item_factors = als_model.item_factors[:n_items]
    tag_model.item_factors = als_model.item_factors[n_items:]
    item_rec_csr = train[:, :n_items]
    tag_rec_csr = train[:, n_items:]
    
    item_ret = []
    tag_ret = []
    from tqdm.auto import tqdm
    for u in tqdm(range(test.shape[0])):
        item_rec = item_model.recommend(u, item_rec_csr, N=1000)
        item_rec = [idx_to_sid[x[0]] for x in item_rec]
        tag_rec = tag_model.recommend(u, tag_rec_csr, N=10) #candiate 용이라 N = 100으로 수정해야함
        tag_rec = [idx_to_tag[x[0]] for x in tag_rec if x[0] in idx_to_tag]
        item_ret.append(item_rec)
        tag_ret.append(tag_rec)
   
    return item_ret, tag_ret