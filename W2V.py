import json
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors

class PlaylistEmbedding:
    def __init__(self, train, val, input_data):
        self.min_count = 3
        self.size = 100
        self.window = 210
        self.sg = 5
        self.p2v_model = WordEmbeddingsKeyedVectors(self.size)
        self.train = train
        self.val = val
        self.input = input_data
        with open('../data/n_results.json', encoding='utf-8') as f:
            self.most_results = json.load(f)
    
    def get_dic(self, train, val):
        song_dic = {}
        tag_dic = {}
        data = train+val
        for q in tqdm(data):
            song_dic[str(q['id'])] = q['songs']
            tag_dic[str(q['id'])] = q['tags']
        self.song_dic = song_dic
        self.tag_dic = tag_dic

    def get_w2v(self, input_data, min_count, size, window, sg):
        w2v_model = Word2Vec(input_data, min_count=min_count, size=size, window=window, sg=sg)
        self.w2v_model = w2v_model

    def update_p2v(self, train, val, w2v_model):
        ID = []
        vec = []
        for q in tqdm(train+val):
            tmp_vec = 0
            if len(q['songs'])>=1:
                for song in q['songs'] + q['tags']:
                    try:
                        tmp_vec += w2v_model.wv_get_vector(str(song))
                    except KeyError:
                        pass
                    