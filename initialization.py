import argparse
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
from elasticsearch import Elasticsearch
from elasticsearch import helpers
global args
import random
import cPickle
from elasticsearch import Elasticsearch
from elasticsearch import helpers

class parse_args():
	def __init__(self, input, output, walkfile, vecinput, directed , num_walks, walk_length):
		self.weighted=False
		self.input = input
		self.output = output
		self.walksile = walkfile
		self.vecinput = vecinput
		self.directed = directed = False
		self.num_walks = num_walks
		self.walk_length = walk_length

#building the randomwalk corpus
def random_walk(G, path_length, alpha=0, rand=random.Random(), start=None):
	""" Returns a truncated random walk.
		path_length: Length of the random walk.
		alpha: probability of restarts.
		start: the start node of the random walk.
	"""
	if start:
		path = [start]
	else:
		path = [rand.choice(list(G.nodes()))]

	while len(path) < path_length:
		cur = path[-1]
		if len(list(G[(cur)])) > 0:
			if rand.random() >= alpha:
				path.append(int(rand.choice(list(G[(cur)]))))
			else:
				path.append(path[0])
		else:
			break
	return [str(node) for node in path]


def build_random_walk_set(G, num_paths, path_length, alpha=0,
					  rand=random.Random(0)):
	walks = []

	nodes = list(G.nodes())

	for cnt in range(num_paths):
		rand.shuffle(nodes)
		for node in nodes:s
			walks.append(random_walk(G, path_length, rand=rand, alpha=alpha, start=node))
	return walks

def elastic_init(walks, ind):
	es_init = Elasticsearch(retry_on_timeout=True)
	mapp = {  		
				"walk": {
				  "properties": {
					"wlks": {
					"type": "text",
					"store": True, 
					"analyzer" : "fulltext_analyzer"		        
					}
				}
			  }
			}
		
	sett = {"settings" : {
		"index" : {
		  "blocks.read_only_allow_delete": False,
		  "number_of_shards" : 5,
		  "number_of_replicas" : 1
		},
		"analysis": {
		  "analyzer": {
			"fulltext_analyzer": {
			  "type": "custom",
			  "tokenizer": "whitespace",
			}
		  }
		}
	  }
	  }

	try:
		es_init.indices.delete(index=ind)
	except:
		pass
	es_init.indices.create(index=ind, body=sett, request_timeout=30)
	es_init.indices.put_mapping(index=ind, doc_type='walk', body=mapp)

	for j in range(0, len(walks)):
		op_dict = {"wlks": " ".join(walks[j])}
		es_init.index(index=ind, doc_type='walk', id=j, body=op_dict)
	es_init.indices.refresh(index=ind)
	return es_init

def learn_embeddings(walks, dimensions, window_size, workers, iteration, output, outputvec, simulatewalks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	if simulatewalks:
		print walks[0]
		model = Word2Vec(walks, size=dimensions, window=window_size, min_count=0, sg=1, workers=workers, iter=iteration)
		model.train(walks, total_examples=model.corpus_count, epochs=model.iter)
		model.save(outputvec)
		model.wv.save_word2vec_format(output + 'main')
	else:
		model = Word2Vec(walks, size=dimensions, window=10, min_count=0, sg=1, workers=1, iter=1)
		model.save(outputvec)
		model.wv.save_word2vec_format(output + 'main')
	model.train(walks, total_examples=model.corpus_count, epochs=model.iter)
	vocab =[]
	keys = []
	for key in sorted(model.wv.vocab, key=lambda x: int(x)):
		keys.append(key)
		vocab.append(np.array(model[key], dtype=float))
	return np.array(vocab), keys



def main(g, indexx, directed, num_walks, walk_length, outputvec, output, dimensions, window_size, workers, iteration, simulatewalks, walkfile):
	print simulatewalks
	if simulatewalks:
		walks = build_random_walk_set(g, num_walks, walk_length, alpha=0, rand=random.Random(0))
		with open(walkfile, 'wb') as pf:
			cPickle.dump(walks, pf)
	else:
		with open(walkfile, 'rb') as pf:
			walks = cPickle.load(pf)
		pf = open(walkfile, 'rb')
		walks = cPickle.load(pf)
		pf.close()
	print len(walks)
	walks = [map(str, walk) for walk in walks]
	lr, keys = learn_embeddings(walks, dimensions, window_size, workers, iteration, output, outputvec, simulatewalks)
	es_init = elastic_init(walks, indexx)
	return walks, es_init, lr, keys

