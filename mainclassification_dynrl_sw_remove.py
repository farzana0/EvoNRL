import argparse
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
import random
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import time

global walks





def random_walk(G, path_length, alpha=0, rand=random.Random(), start=None):
    """ Returns a truncated random walk.
        path_length: Length of the random walk.
        alpha: probability of restarts.
        start: the start node of the random walk.
    """
    if start:
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(list(G.keys()))]

    while len(path) < path_length:
      cur = path[-1]
      if len(G[cur]) > 0:
        if rand.random() >= alpha:
          path.append(rand.choice(list(G[cur])))
        else:
          path.append(path[0])
      else:
        break
    return [str(node) for node in path]




def learn_embeddings(walks, inputvec, output ):


	model = Word2Vec(walks, seed=0, size=128, window=10, min_count=0, sg=1, workers=1, iter=1)
	model.train(walks, total_examples=model.corpus_count, epochs=model.iter)
	print model.corpus_count
	model.wv.save_word2vec_format(output)
	vocab =[]
	keys=[]
	for key in sorted(model.wv.vocab, key=lambda x: int(x)):
		keys.append(key)
		vocab.append(np.array(model[key], dtype=float))
	return np.array(vocab), keys


	



def walks_increment(G, walklength, walks, es, edges, ind):
	ccc = 0
	change=0
	for i in edges:	
		ccc=ccc+1
		G.remove_edge(*i)
		node_i = str(i[0])
		node_j = str(i[1])
		es.indices.refresh(index=ind)
		res_i = helpers.scan(client=es, query={"query": {"match_phrase": {"wlks": {"query": node_i + " " + node_j}}}}, index = ind, size = 10000, scroll='1m')
		degree_i = G.degree(i[0])
		degree_j = G.degree(i[1])
		
		blk=[]
		for itemed in res_i:
			itemed_0 = int(itemed['_id'])
			itemed_1 = (es.termvectors(index=ind, doc_type='walk', id=itemed_0, field_statistics=False, term_statistics=False, offsets=False, positions=True, fields = ['wlks'])['term_vectors']['wlks']['terms'][node_i]['tokens'][0]['position'])
			change=int(itemed_1)%100
			#node deletion change
			if degree_i == 0 & change == 0:
				action = {
											"_op_type": 'delete',
								 			"_index": ind,
											"_type": "walk",
											"_id": itemed_0
				 									}
				blk.append(action)
				helpers.bulk(es, blk)
				es.indices.refresh(index=ind)
				blk = []
				#print 'delete'
				#print itemed_0
			elif degree_i == 0 & change !=0:
				walks[itemed_0][(change):] = random_walk(G, walklength-change, alpha=0, rand=random.Random(), start=i[1])
				action = {
											"_op_type": 'update',
								 			"_index": ind,
											"_type": "walk",
											"_id": itemed_0,
											"_source": {"doc":{
											"wlks": " ".join(walks[itemed_0])
																}
														}
				 									}
				blk.append(action)	
				helpers.bulk(es, blk)
				es.indices.refresh(index=ind)
				blk = []
			else:
				walks[itemed_0][(change):] = random_walk(G, walklength-change, alpha=0, rand=random.Random(), start=i[0])
				action = {
											"_op_type": 'update',
								 			"_index": ind,
											"_type": "walk",
											"_id": itemed_0,
											"_source": {"doc":{
											"wlks": " ".join(walks[itemed_0])
																}
														}
				 									}
				blk.append(action)	
				helpers.bulk(es, blk)
				es.indices.refresh(index=ind)
				blk = []
		es.indices.refresh(index=ind)
		res_j = helpers.scan(client=es, query={"query": {"match_phrase": {"wlks": {"query": node_j + " " + node_i}}}}, index = ind, size = 10000, scroll='1m')
		es.indices.refresh(index=ind)

		for itemed in res_j:
			itemed_0 = int(itemed['_id'])
			itemed_1 = (es.termvectors(index=ind, doc_type='walk', id=itemed_0, field_statistics=False, term_statistics=False, offsets=False, positions=True, fields = ['wlks'])['term_vectors']['wlks']['terms'][node_j]['tokens'][0]['position'])
			change=int(itemed_1)%100
			if degree_j == 0 & change == 0:
				action = {
											"_op_type": 'delete',
								 			"_index": ind,
											"_type": "walk",
											"_id": itemed_0
				 									}
				blk.append(action)
				helpers.bulk(es, blk)
				es.indices.refresh(index=ind)
				blk = []
				#print 'delete'
				#print itemed_0
			elif degree_j == 0 & change != 0:
				walks[itemed_0][int(change):] = random_walk(G, walklength-change, alpha=0, rand=random.Random(), start=i[0])
				action = {
											"_op_type": 'update',
								 			"_index": ind,
											"_type": "walk",
											"_id": itemed_0,
											"_source": {"doc":{
											"wlks": " ".join(walks[itemed_0])
																}
														}
				 									}
				blk.append(action)	
				helpers.bulk(es, blk)
				es.indices.refresh(index=ind)
				blk = []
			else:
				walks[itemed_0][int(change):] = random_walk(G, walklength-change, alpha=0, rand=random.Random(), start=i[1])
				action = {
												"_op_type": 'update',
								 			"_index": ind,
											"_type": "walk",
											"_id": itemed_0,
											"_source": {"doc":{
											"wlks": " ".join(walks[itemed_0])
																}
														}
				 									}
				blk.append(action)	

				helpers.bulk(es, blk)
				es.indices.refresh(index=ind)
				blk = []
		
	print 'updated!'
	return walks


def main(g, walks, es, edges, wl, num, ind, inputvec, output):
	print (len(list(nx.isolates(g))))
	walks_new = walks_increment(g, wl, walks, es, edges, ind)
	lr, keys = learn_embeddings(walks_new, inputvec, output)
	return lr, keys, walks_new

	










