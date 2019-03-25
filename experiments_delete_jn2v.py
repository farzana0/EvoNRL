import numpy as np
import networkx as nx
from gensim.models import Word2Vec
import node_addition_main_sameinits
import time
import random
from statistics import mean
import matplotlib.pyplot as plt
import os
import cPickle
import mainclassification_sameinits as ms
import mainclassification_dynrl 
import multilabelClassification as mc
import learn_embedding_node2vec as lem
import operator
import numpy as np
import csv
import mainclassification_dynrl_iji 
#import mainclassification_dynrl_sw_remove as md
import general_update as md
#import evonrlplus_al as md
import os
import argparse
import scipy.io as sio
import cPickle
from elasticsearch import Elasticsearch
from elasticsearch import helpers

def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run node2vec.")

	parser.add_argument('--input', nargs='?', default='edges.csv',
	                    help='Input graph path')

	parser.add_argument('--vecinput', nargs='?', default='node2vec_sameinits/inits/inits_bc.vec',
	                    help='Embeddings path')
	parser.add_argument('--output', nargs='?', default='sample.emb',
	                    help='Embeddings path')
	parser.add_argument('--sampleoutput', nargs='?', default='sample.txt',
	                    help='sampleoutput')
	parser.add_argument('--vecoutput', nargs='?', default='node2vec_sameinits/inits/inits_bc.vec',
	                    help='Embeddings path')

	parser.add_argument('--walkfile', nargs='?', default='evonrlsw_walks_v0.pkl',
	                    help='Embeddings path')
	parser.add_argument('--simulatewalks', type=bool, default=False,
	                    help='if true simulates new walks else reads the walkfile IN CASE of FALSE DO NOT INPUT' )

	parser.add_argument('--indexx', type=str, default='index_variable',
	                    help='index')

	parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=80,
	                    help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=1, type=int,
                      help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=8,
	                    help='Number of parallel workers. Default is 8.')

	parser.add_argument('--p', type=float, default=1,
	                    help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=1,
	                    help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--repeat', type=int, default=2,
	                    help='numper of repetition')
	parser.add_argument('--precfile', type=str, default='prec_save_bc.pkl',
	                    help='results')

	parser.add_argument('--ppi', type=bool, default=False,
	                    help='is this ppi. IN CASE OF FALSE DO NOT INPUT')
	parser.add_argument('--labelfile', type=str, default='labels.pkl',
	                    help='is this ppi.')
	parser.add_argument('--totalsteps', type=int, default=10,
	                    help='is this totaledges.')
	parser.add_argument('--sample', type=int, default=2000,
	                    help='sampling number.')
	parser.add_argument('--bunch', type=int, default=2000,
	                    help='is this bunch.')
	parser.add_argument('--remove', type=bool, default=False,
	                    help='is this bunch. IN CASE oF FALSE DO NOT INPUT')
	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--n2vprecfile', type=str, default='n2vprec_save_bc.pkl',
	                    help='results')
	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)


	return parser.parse_args()

def sampling():
	if args.ppi:
		g0 = nx.read_edgelist(args.input, nodetype=int)
	else:
		g0 = nx.read_edgelist(args.input, delimiter=',', nodetype=int)
	for edge in g0.edges():
		g0[edge[0]][edge[1]]['weight'] = 1
	all_edges = list(g0.edges())
	all_nodes = list(g0.nodes())
	#g = nx.read_edgelist(args.input, nodetype=int)
	#print g.number_of_nodes()
	#print g.number_of_edges()
	
	samp = random.sample(range(10312), args.sample)
	g = nx.Graph(g0.subgraph(sorted([list(g0.nodes())[i] for i in samp])))
	current_nodes = list(g.nodes())
	current_edges = list(g.edges())
	with open("sample_nodeaddition.edgelist",'wb') as f:
		nx.write_edgelist(g, f)
	potential_edges = list(set(all_edges) - set(current_edges))
	potential_nodes = list(set(all_nodes) - set(current_nodes))
	print len(potential_edges)
	print g.number_of_nodes()
	print g.number_of_edges()
	a = (list(nx.isolates(g)))
	iso = [list(g0.nodes()).index(i) for i in a]
	samp = list(set(samp) - set(iso))
	'''for edge in g.edges():
		g[edge[0]][edge[1]]['weight'] = 1'''
	return g, g0

def getlabels(samp):
	if args.ppi:
		mat = sio.loadmat(args.labelfile)
  		labels = mat['group'].todense()
  		labels = np.array(labels)
	else:
		with open(args.labelfile, 'rb') as fp:
			labels =cPickle.load(fp)
	labels=labels[samp]
	#labels = np.loadtxt(args.labelfile)
	return labels


def justnode2vec(g, labels, prec_node2vec):
	ls =[]
	for ii in range(args.repeat):
		lr0 = lem.eval(p=1, q=1, g=g, sd=ii, walks_0=[], i=1)
		ls.append(mc.averageprec(lr0, labels))
	print ls
	prec_node2vec.append(ls)
	prec_node2vec.append(mc.averageprec(lr0, labels))
	with open(args.n2vprecfile, 'wb') as fp:
		cPickle.dump(prec_node2vec, fp)
	return prec_node2vec
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def edgeloops(g, walks, j, g0):
	prec_node2vec = []
	prec_node2vec_dyn = []
	
	with open('sampleandsave_edges_1000_ppi.pkl', 'rb') as pf:
		ebunch = cPickle.load( pf)
	ls = []
	for ii in range(args.repeat):
		lr0, keys = lem.eval(p=1, q=1, g=g, sd=0, walks_0=[], i=1)
		samp = [list(g0.nodes()).index(int(i)) for i in keys]
		print keys[:10]
		print samp[:10]
		labels = getlabels(samp)
		ls.append(mc.averageprec(lr0, labels))
		print ls
	for ebunc in reversed(ebunch):
		if args.remove:
			g.remove_edges_from(ebunc)
		else:
			g.add_edges_from(ebunc)
		#g.remove_nodes_from(list(nx.isolates(g)).copy())
		print 'ebunch'
		print len(ebunc)
		print 'number of edges before addition'
		print g.number_of_edges()
		print 'number of nodes before addition'
		print g.number_of_nodes()
		ls =[]
		for ii in range(args.repeat):
			lr0, keys = lem.eval(p=1, q=1, g=g, sd=0, walks_0=[], i=1)
			samp = [list(g0.nodes()).index(int(i)) for i in keys]
			labels = getlabels(samp)
			ls.append(mc.averageprec(lr0, labels))
		print 'number of nodes'
		print len(samp)
		print 'number of edges before addition'
		print g.number_of_edges()
		print ls
		prec_node2vec.append(ls)
		prec_node2vec.append(mean(ls))
		with open(args.n2vprecfile, 'wb') as fp:
			cPickle.dump(prec_node2vec, fp)
		
		print 'after node addition'
		print g.number_of_edges()
		print g.number_of_nodes()
	return prec_node2vec
def graphs():
	if args.ppi:
		g0 = nx.read_edgelist(args.input, nodetype=int)
	else:
		g0 = nx.read_edgelist(args.input, delimiter=',', nodetype=int)
	for edge in g0.edges():
		g0[edge[0]][edge[1]]['weight'] = 1
	g = nx.read_gpickle("sample_nodeaddition_ppi.gpickle")
	for edge in g0.edges():
		g0[edge[0]][edge[1]]['weight'] = 1
	print g.number_of_nodes()
	print g.number_of_edges()
	#print len(g.isolates())
	g = g0.copy()
	return g, g0

def learn_embeddings(walks_0, sd):
	
	walks_0 = [map(str, walk) for walk in walks_0]
	model = Word2Vec(walks_0, seed=sd, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
	model.train(walks_0, total_examples=model.corpus_count, epochs=model.iter)
	#model.wv.save_word2vec_format('outputtest.emb')
	vocab =[]
	keys = []
	for key in sorted(model.wv.vocab, key=lambda x: int(x)):
		keys.append(key)
		
		vocab.append(np.array(model[key], dtype=float))
	return np.array(vocab), keys

def main(args):
	
	g, g0 = graphs()
	#labels = getlabels()
	#walks, es, vocab = main_sameinits_es.main(g, args.indexx, args.directed, args.num_walks, args.walk_length, args.vecinput, args.output, args.dimensions, args.window_size, args.workers, args.iter, args.simulatewalks, args.walkfile)
	#print mc.averageprec(vocab, labels)
	'''if args.simulatewalks:
		with open(args.walkfile, 'wb') as fp:
			cPickle.dump(walks, fp)'''
	#prec_node2vec=(mc.averageprec(vocab, labels)) 
	#print mc.averageprec(vocab, labels)
	print len(list(g.nodes()))
	with open('ppi_500_sample_walks.pkl', 'rb') as fp:
		walks = cPickle.load(fp)
	lr0, keys = learn_embeddings(walks, sd=0)
	samp = [list(g0.nodes()).index(int(i)) for i in keys]
	labels = getlabels(samp)
	print mc.averageprec(lr0, labels)
	for j in range(1):
		
		newindex = 'test' 
		
		prec_node2vec = edgeloops(g, walks, j, g0)
		print prec_node2vec


if __name__ == "__main__":
	args = parse_args()
	main(args)