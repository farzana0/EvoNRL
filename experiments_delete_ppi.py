import numpy as np
import networkx as nx
from gensim.models import Word2Vec
import node_deletion_main_sameinits
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
import mainclassification_dynrl_sw_remove as md
import os
import argparse
import scipy.io as sio

def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run node2vec.")

	parser.add_argument('--input', nargs='?', default='edges.csv',
	                    help='Input graph path')

	parser.add_argument('--vecinput', nargs='?', default='node2vec_sameinits/inits/inits_samp.vec',
	                    help='Embeddings path')
	parser.add_argument('--output', nargs='?', default='sample.emb',
	                    help='Embeddings path')
	parser.add_argument('--sampleoutput', nargs='?', default='sample.txt',
	                    help='sampleoutput')
	parser.add_argument('--vecoutput', nargs='?', default='node2vec_sameinits/inits/inits_samp.vec',
	                    help='Embeddings path')

	parser.add_argument('--walkfile', nargs='?', default='emb/walks.txt',
	                    help='Embeddings path')
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

	
	parser.add_argument('--precfile', type=str, default='prec_save_bc.pkl',
	                    help='results')
	parser.add_argument('--n2vprecfile', type=str, default='n2vprec_save_bc.pkl',
	                    help='results')

	parser.add_argument('--repeat', type=int, default=2,
	                    help='numper of repetition')

	parser.add_argument('--ppi', type=bool, default=False,
	                    help='is this ppi.')
	parser.add_argument('--labelfile', type=str, default='labels.pkl',
	                    help='is this ppi.')
	parser.add_argument('--totaledges', type=int, default=10000,
	                    help='is this totaledges.')
	parser.add_argument('--bunch', type=int, default=2000,
	                    help='is this bunch.')
	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)


	return parser.parse_args()

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def sampling():
	if args.ppi:
		g = nx.read_edgelist(args.input, nodetype=int)
	else:
		g = nx.read_edgelist(args.input, delimiter=',', nodetype=int)
	'''
	samp = random.sample(range(args.sample), args.sample)
	g = nx.Graph(g0.subgraph(sorted([list(g0.nodes)[i] for i in samp])))
	a = (list(nx.isolates(g)))
	iso = [list(g0.nodes()).index(i) for i in a]
	samp = list(set(samp) - set(iso))
	'''
	#nx.write_edgelist(g, args.sampleoutput)
	with open("sample_nodeaddition.edgelist",'wb') as f:
		nx.write_edgelist(g, f)
	g0 = nx.read_edgelist("sample_nodeaddition.edgelist")
	for edge in g0.edges():
		g0[edge[0]][edge[1]]['weight'] = 1
	for edge in g.edges():
		g[edge[0]][edge[1]]['weight'] = 1
	return g

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

'''def getlabels():
	if args.ppi:
		mat = sio.loadmat(args.labelfile)
  		labels = mat['group'].todense()
  		labels = np.array(labels)
	else:
		with open(args.labelfile, 'rb') as fp:
			labels =cPickle.load(fp)
		#labels=labels[samp]
	return labels'''

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

def justnode2vec(g, labels):
	ls =[]
	prec_node2vec = []
	for ii in range(args.repeat):
		lr0 = lem.eval(p=1, q=1, g=g, sd=ii, walks_0=[], i=1)
		ls.append(mc.averageprec(lr0, labels))
	print ls
	prec_node2vec.append(ls)
	prec_node2vec.append(mc.averageprec(lr0, labels))
	with open(args.n2vprecfile, 'wb') as fp:
		cPickle.dump(prec_node2vec, fp)

def edgeloops(g, walks, es, g0):
	prec_node2vec_dyn = []
	with open('sampleandsave_edges_1000_ppi.pkl', 'rb') as pf:
		ebunch = cPickle.load( pf)
	print len(ebunch)

	for ebunc in reversed(ebunch):
		print len(ebunc)
		print 'number of edges'
		print g.number_of_edges()
		#print i
		#ebunch = random.sample(list(nx.edges(g)), i)
		lr, keys, walks = md.main(g, walks, edges=ebunc, es=es, wl=80, num=0, ind=args.indexx, inputvec=args.vecinput, output=args.output)
		g.remove_edges_from(ebunc)
		#g.remove_nodes_from(list(nx.isolates(g)).copy())
		a = (list(keys))
		samp = [list(g0.nodes()).index(int(i)) for i in a]
		#justnode2vec(g, getlabels(samp))
		print 'number of edges'
		print g.number_of_edges()
		print 'number of nodes'
		print len(samp)
		prec_node2vec_dyn.append(mc.averageprec(lr, getlabels(samp)))
		print mc.averageprec(lr, getlabels(samp))
		with open(args.precfile, 'wb') as fp:
			cPickle.dump(prec_node2vec_dyn, fp)
		print prec_node2vec_dyn
	return prec_node2vec_dyn




def main(args):

	g, g0 = graphs()
	#labels = getlabels()
	walks, es, vocab, keys = node_deletion_main_sameinits.main(g, args.indexx , args.directed, args.num_walks, args.walk_length, args.vecinput, args.output, args.dimensions, args.window_size, args.workers, args.iter, True , args.walkfile)
	#prec_node2vec=(mc.averageprec(vocab, labels)) 
	#print mc.averageprec(vocab, labels)
	a = (list(keys))
	samp = [list(g0.nodes()).index(int(i)) for i in a]
	#justnode2vec(g, getlabels(samp))
	print mc.averageprec(vocab, getlabels(samp))
	wk = walks
	prec_node2vec_dyn = edgeloops(g, walks, es, g0)
	print prec_node2vec_dyn


if __name__ == "__main__":
	args = parse_args()
	main(args)