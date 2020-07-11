from gensim.models import Word2Vec
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors

class PlaylistEmbedding:
    def __init__(self):
        self.min_count = 3
        self.size = 100
        self.window = 210
        self.sg = 