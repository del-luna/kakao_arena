from tqdm import tqdm
import numpy as np
from scipy.sparse import *

from implicit.evaluation import  *
from implicit.als import AlternatingLeastSquares as ALS

#def als_run(data, n_items, train, test, idx_to_sid, idx_to_tag):
def als_run(data, n_items):
    als_model = ALS(factors=128, regularization=0.08)
    als_model.fit(data.T * 15.0) #item-user matrix -> transpose
    item_model = ALS(use_gpu=False)
    tag_model = ALS(use_gpu=False)
    item_model.user_factors = als_model.user_factors
    tag_model.user_factors = als_model.user_factors
    item_model.item_factors = als_model.item_factors[:n_items]
    tag_model.item_factors = als_model.item_factors[n_items:]
    item_rec_csr = data[:, :n_items]
    tag_rec_csr = data[:, n_items:]
    
    item_list = []
    tag_list = []
    for u in tqdm(range(data.shape[0])):
        item_rec = item_model.recommend(u, item_rec_csr, N=7000)
        tag_rec = tag_model.recommend(u, tag_rec_csr, N=100)
        item_list.append(item_rec)
        tag_list.append(item_rec)
    
    return item_list, tag_list
    '''
    item_ret = []
    tag_ret = []
    from tqdm.auto import tqdm
    for u in tqdm(range(test.shape[0])):
        item_rec = item_model.recommend(u, item_rec_csr, N=5000)
        item_rec = [idx_to_sid[x[0]] for x in item_rec]
        tag_rec = tag_model.recommend(u, tag_rec_csr, N=100)
        tag_rec = [idx_to_tag[x[0]] for x in tag_rec if x[0] in idx_to_tag]
        item_ret.append(item_rec)
        tag_ret.append(tag_rec)
    
    return als_model, item_model, tag_model
    '''