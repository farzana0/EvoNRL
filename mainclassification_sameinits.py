import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
import pickle



class parse_args():
	def __init__(self, p, q):
		#self.input = 'graph/PPIedgelist.txt'
		self.output = 'emb/PPIembedding.emb'
		self.dimensions = 128
		self.walk_length = 80
		self.num_walks = 10
		self.window_size = 10
		self.iter = 1
		self.workers = 8
		self.weighted = False
		self.directed = False
		self.p = p
		self.q = q

def elastic_init(walks):
	es = Elasticsearch()
	mapp = {  		
			    "walk": {
			    "dynamic": "strict",
			      "properties": {
			        "wlks": {
			        "type": "string",
			        "store": True, 
			        "index": "analyzed",
			        "index_options": "positions",
			        "term_vector": "with_positions_offsets",
			        "analyzer" : "fulltext_analyzer"		        
			        }
			    }
			  }
			}
		
	sett = {"settings" : {
	    "index" : {
	      "number_of_shards" : 1,
	      "number_of_replicas" : 0
	    },
	    "analysis": {
	      "analyzer": {
	        "fulltext_analyzer": {
	          "type": "custom",
	          "tokenizer": "whitespace",
	          "filter": [
	            "lowercase",
	            "type_as_payload"
	          ]
	        }
	      }
	    }
	  }
	  }

	es.indices.delete(index='test_indi', ignore=[400, 404])
	es.indices.create(index="test_indi", body=sett)
	es.indices.put_mapping(index="test_indi", doc_type='walk', body=mapp)

	for j in range(0, len(walks)):
		op_dict = {"wlks": walks[j]}
		es.index(index='test_indi', doc_type='walk', id=j, body=op_dict)
	es.indices.refresh(index="test_indi")
	return es


def learn_embeddings(walks):

	walks = [map(str, walk) for walk in walks]
	model = Word2Vec.load(args.vecinput)
	model.train(walks, total_examples=model.corpus_count, epochs=model.iter)
	model.wv.save_word2vec_format(args.output+ 'mainclassification')
	model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
	
	vocab =[]
	for key in sorted(model.wv.vocab, key=lambda x: int(x)):
		
		vocab.append(np.array(model[key], dtype=float))

	return np.array(vocab)

def main(args, nx_G):
	G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
	G.preprocess_transition_probs()
	walks = G.simulate_walks(args.num_walks, args.walk_length)
	lr = learn_embeddings(walks)
	es = elastic_init(walks)
	return walks, es

def eval(p,q,g):
	global args
	args = parse_args(p,q)
	return main(args, g)
