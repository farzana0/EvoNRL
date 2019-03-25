import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
from elasticsearch import Elasticsearch
from elasticsearch import helpers
global args
import random
import cPickle
from elasticsearch import Elasticsearch
from elasticsearch import helpers

class parse_args():
	def __init__(self, input, output, walkfile, vecinput, directed , p, q, num_walks, walk_length):
		self.weighted=False
		self.input = input
		self.output = output
		self.walksile = walkfile
		self.vecinput = vecinput
		self.directed = directed = False
		self.p = p = 1
		self.q = q = 1
		self.num_walks = num_walks
		self.walk_length = walk_length

def elastic_init(walks, ind):
	es_init = Elasticsearch(retry_on_timeout=True)
	mapp = {  		
			    "walk": {
			    #"type": "string",
			      "properties": {
			        "wlks": {
			        "type": "text",
			        #"dynamic": True,
			        "store": True, 
			        #"index": "not_analyzed",
			        #"index_options": "positions",
			        #"term_vector": "with_positions_offsets",
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
	         # "filter": [
	          #  "lowercase",
	           # "type_as_payload"
	          #]
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
	'''snapshot_body ={"type": "fs",
    "settings": {
    "location": "/home/farzanah/elasticsearch-backup/my_backup_location"}}
	try:
		es_init.snapshot.delete(repository='backup', snapshot='ppi_walks_node_addition')
	except:
 		pass
	es_init.snapshot.create_repository(repository='backup', body=snapshot_body)
	es_init.snapshot.create(repository='backup', snapshot='ppi_walks_node_addition', wait_for_completion=True)
	#print es_init.indices.stats(index="ppitest")
	print len(list(helpers.scan(client=es_init, query={"query": {"match_all":{}}}, index = "ppitest", size = 10000, scroll='1m')))
	#print es_init.snapshot.status(repository='backup', snapshot='bc_walks_v1')
	es_init.indices.close(index='ppitest')
	es_init.snapshot.restore(repository='backup', snapshot='ppi_walks_node_addition', body={
		"settings": {
		 "indices": "ppitest",
		"location": "/home/farzanah/elasticsearch-backup/my_backup_location"}}, wait_for_completion=True)'''
		#print mc.averageprec(vocab, labels)
	#print es_init.indices.stats(index="ppitest")
	#print len(list(helpers.scan(client=es_init, query={"query": {"match_all":{}}}, index = "ppitest", size = 10000, scroll='1m')))
	return es_init

def read_graph(g):
	'''
	Reads the input network in networkx.
	'''

	for i in g.nodes():
		stri = str(i)
		op_dict = {stri: g.degree(i)}

	return op_dict



	return G

def learn_embeddings(walks, dimensions, window_size, workers, iteration, output, outputvec, simulatewalks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	#walks = [map(str, walk) for walk in walks]

	if simulatewalks:
		model = Word2Vec(walks, size=128, window=window_size, min_count=0, sg=1, workers=workers, iter=iteration)
		model.train(walks, total_examples=model.corpus_count, epochs=model.iter)
		model.save(outputvec)
		model.wv.save_word2vec_format(output + 'main')
	else:
		model = Word2Vec(walks, size=128, window=10, min_count=0, sg=1, workers=1, iter=1)
		model.train(walks, total_examples=model.corpus_count, epochs=model.iter)
		model.save(outputvec)
		model.wv.save_word2vec_format(output + 'main')
	vocab =[]
	keys = []
	for key in sorted(model.wv.vocab, key=lambda x: int(x)):
		vocab.append(np.array(model[key], dtype=float))
		keys.append(key)
	return np.array(vocab), keys



def main(g, indexx, directed, num_walks, walk_length, outputvec, output, dimensions, window_size, workers, iteration, simulatewalks, walkfile):
	#dic = read_graph(g)
	nx_G = g
	p=1
	q=1
	print simulatewalks
	if simulatewalks:
		G = node2vec.Graph(nx_G, directed, p, q)
		G.preprocess_transition_probs()
		walks = G.simulate_walks(num_walks, walk_length)
		with open(walkfile, 'wb') as pf:
			cPickle.dump(walks, pf)
	else:
		'''with open(walkfile, 'rb') as pf:
			walks = cPickle.load(pf)'''
		pf = open(walkfile, 'rb')
		walks = cPickle.load(pf)
		pf.close()

	walks = [map(str, walk) for walk in walks]
	lr, keys = learn_embeddings(walks, dimensions, window_size, workers, iteration, output, outputvec, simulatewalks)

	es_init = elastic_init(walks, indexx)

	return walks, es_init, lr, keys


#args = parse_args(input = 'edges.csv', output = 'ppi_inc.emb', walk_length = 80, num_walks = 10, walkfile = 'node2vec_dyn/walk/walks.txt', vecinput= 'node2vec_sameinits/inits/inits_samp.vec', p=1, q=1, directed=False)
