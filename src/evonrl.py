import argparse
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
import random
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import time
import cPickle
global args
import time
import copy


class parse_args():
	def __init__(self, input, output, walkfile, vecinput, directed ,  num_walks, walk_length):
		self.weighted=False
		self.output = output
		self.walksile = walkfile
		self.vecinput = vecinput
		self.directed = directed = False
		self.num_walks = num_walks
		self.walk_length = walk_length



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




def learn_embeddings(walks_new, inputvec, output, edges):
	model = Word2Vec(walks_new, size=128, window=10, min_count=0, sg=1, workers=1, iter=1)	
	model.train(walks_new, total_examples=model.corpus_count, epochs=model.iter)
	model.wv.save_word2vec_format(output + 'emb')
	vocab =[]
	keys=[]
	for key in sorted(model.wv.vocab, key=lambda x: int(x)):
		keys.append(key)
		vocab.append(np.array(model[key], dtype=float))
	return np.array(vocab), keys
	

def addition(G, num_walks, walklength, walks, es, edge, ind):
	G.add_edge(*edge, weight=1)
	node_i = str(edge[0])
	node_j = str(edge[1])
	allwalks_i = helpers.scan(client=es, query={"query": {"match_phrase": {"wlks": {"query": node_i}}}}, index = ind, size = 10000, scroll='1m')
	allwalks_i = list(allwalks_i)
	degree_i = G.degree(edge[0])
	degree_j = G.degree(edge[1])
	sampled_i = random.sample(allwalks_i, int(len(allwalks_i)/degree_i))
	blk = []
	if len(sampled_i) > 0:
		for wk in sampled_i:
			wk_id = int(wk['_id'])
			wk_pos = random.choice(es.termvectors(index=ind, doc_type='walk', id=wk_id, field_statistics=False, term_statistics=False, offsets=False, positions=True, fields = ['wlks'])['term_vectors']['wlks']['terms'][node_i]['tokens'])['position']
			wk_pos = wk_pos % 100
			walks[wk_id][wk_pos+1:] = random_walk(G, walklength-wk_pos-1, alpha=0, rand=random.Random(), start=edge[1])
			action = {
							"_op_type": 'update',
						 	"_index": ind,
							"_type": "walk",
							"_id": int(wk['_id']),
							"_source": {"doc" : {
							"wlks": ' '.join((walks[wk_id]))
												}
										}
		 							}
	 		blk.append(action)
		helpers.bulk(es, blk)
	es.indices.refresh(index=ind)
	allwalks_j = helpers.scan(client=es, query={"query": {"match_phrase": {"wlks": {"query": node_j}}}}, index=ind,
						 size=10000, scroll='1m')
	allwalks_j = list(allwalks_j)
	sampled_j = random.sample(allwalks_j, int(len(allwalks_j)/degree_j))
	blk=[]
	if len(sampled_j)> 0:
		for wk in sampled_j:
			wk_id = int(wk['_id'])
			wk_pos = random.choice(es.termvectors(index=ind, doc_type='walk', id=wk_id, field_statistics=False, term_statistics=False, offsets=False, positions=True, fields = ['wlks'])['term_vectors']['wlks']['terms'][node_j]['tokens'])['position']
			wk_pos = wk_pos % 100
			walks[wk_id][wk_pos+1:] = random_walk(G, walklength-wk_pos-1, alpha=0, rand=random.Random(), start=edge[0])
			action = {
									"_op_type": 'update',
						 			"_index": ind,
									"_type": "walk",
									"_id": int(wk['_id']),
									"_source": {"doc":{
									"wlks": ' '.join((walks[wk_id]))															}
													}
			 								}
							
			blk.append(action)
		helpers.bulk(es, blk)

	blk = []
	if len(allwalks_i) == 0:
		print 'node addition'
		for kk in range(num_walks):
			random.seed(kk)
			walks.append(random_walk(G, walklength, alpha=0, rand=random.Random(), start=edge[0]))
			action = {
									"_op_type": 'create',
						 			"_index": ind,
									"_type": "walk",
									"_id": len(walks),
									"_source": {"doc":{
									"wlks": ' '.join((walks[-1]))
														}
												}
		 									}
							
		 	blk.append(action)

				
	if len(allwalks_j) == 0:
		print 'node addition'
		for kk in range(num_walks):
			random.seed(kk)
			walks.append(random_walk(G, walklength, alpha=0, rand=random.Random(), start=edge[1]))
			action = {
									"_op_type": 'create',
						 			"_index": ind,
									"_type": "walk",
									"_id": len(walks),
									"_source": {"doc":{
									"wlks": ' '.join((walks[-1]))
														}
												}
		 									}
						
		 	blk.append(action)
	helpers.bulk(es, blk)
	es.indices.refresh(index=ind)
	print 'updated!'
	return walks, G


def deletion(G, num_walks,  walklength, walks, es, edge, ind):
	G.remove_edge(*edge)
	node_i = str(edge[0])
	node_j = str(edge[1])
	es.indices.refresh(index=ind)
	allwalks_i = helpers.scan(client=es, query={"query": {"match_phrase": {"wlks": {"query": node_i + " " + node_j}}}}, index = ind, size = 10000, scroll='1m')
	degree_i = G.degree(edge[0])
	degree_j = G.degree(edge[1])
	blk=[]
	for wk in allwalks_i:
		wk_id = int(wk['_id'])
		wk_pos = (es.termvectors(index=ind, doc_type='walk', id=wk_id, field_statistics=False, term_statistics=False, offsets=False, positions=True, fields = ['wlks'])['term_vectors']['wlks']['terms'][node_i]['tokens'][0]['position'])
		change=int(wk_pos)%100
		#node deletion change
		if degree_i == 0 & change == 0:
			action = {
										"_op_type": 'delete',
							 			"_index": ind,
										"_type": "walk",
										"_id": wk_id
			 									}
			blk.append(action)
			helpers.bulk(es, blk)
			es.indices.refresh(index=ind)
			blk = []
		elif degree_i == 0 & change !=0:
			walks[wk_id][(change):] = random_walk(G, walklength-change, alpha=0, rand=random.Random(), start=i[1])
			action = {
										"_op_type": 'update',
							 			"_index": ind,
										"_type": "walk",
										"_id": wk_id,
										"_source": {"doc":{
										"wlks": " ".join(walks[wk_id])
															}
													}
			 									}
			blk.append(action)	
			helpers.bulk(es, blk)
			es.indices.refresh(index=ind)
			blk = []
		else:
			walks[wk_id][(change):] = random_walk(G, walklength-change, alpha=0, rand=random.Random(), start=i[0])
			action = {
										"_op_type": 'update',
							 			"_index": ind,
										"_type": "walk",
										"_id": wk_id,
										"_source": {"doc":{
										"wlks": " ".join(walks[wk_id])
															}
													}
			 									}
			blk.append(action)	
			helpers.bulk(es, blk)
			es.indices.refresh(index=ind)
			blk = []
	es.indices.refresh(index=ind)
	allwalks_j = helpers.scan(client=es, query={"query": {"match_phrase": {"wlks": {"query": node_j + " " + node_i}}}}, index = ind, size = 10000, scroll='1m')
	for wk in allwalks_j:
		wk_id = int(wk['_id'])
		wk_pos = (es.termvectors(index=ind, doc_type='walk', id=wk_id, field_statistics=False, term_statistics=False, offsets=False, positions=True, fields = ['wlks'])['term_vectors']['wlks']['terms'][node_j]['tokens'][0]['position'])
		change=int(wk_pos)%100
		if degree_j == 0 & change == 0:
			action = {
										"_op_type": 'delete',
							 			"_index": ind,
										"_type": "walk",
										"_id": wk_id
			 									}
			blk.append(action)
			helpers.bulk(es, blk)
			es.indices.refresh(index=ind)
			blk = []
		elif degree_j == 0 & change != 0:
			walks[wk_id][int(change):] = random_walk(G, walklength-change, alpha=0, rand=random.Random(), start=i[0])
			action = {
											"_op_type": 'update',
							 			"_index": ind,
										"_type": "walk",
										"_id": wk_id,
										"_source": {"doc":{
										"wlks": " ".join(walks[wk_id])
															}
													}
			 									}
			blk.append(action)	
			helpers.bulk(es, blk)
			es.indices.refresh(index=ind)
			blk = []
		else:
			walks[wk_id][int(change):] = random_walk(G, walklength-change, alpha=0, rand=random.Random(), start=i[1])
			action = {
											"_op_type": 'update',
							 			"_index": ind,
										"_type": "walk",
										"_id": wk_id,
										"_source": {"doc":{
										"wlks": " ".join(walks[wk_id])
															}
													}
			 									}
			blk.append(action)	
			helpers.bulk(es, blk)
			es.indices.refresh(index=ind)
			blk = []		
	return walks, G



def walks_update(G, num_walks, walklength, walks, es, edges, ind):
	for edge in edges:
		print edge[1]
		if edge[1] == '1':
			walks, G = addition(G, num_walks, walklength, walks, es, edge[0], ind)	
		else:
			walks, G = deletion(G, num_walks,  walklength, walks, es, edge[0], ind)
	return walks_new

		


def main(g, walks, num_walks, es, edges,wl, ind, inputvec, output):
	walks_new = walks_update(g, num_walks, wl, walks, es, edges, ind)
	with open(output + 'walks.pkl', 'wb') as pf:
			cPickle.dump(walks_new, pf)
	lr, keys = learn_embeddings(walks_new, inputvec, output, edges)
	return lr, walks_new, keys




