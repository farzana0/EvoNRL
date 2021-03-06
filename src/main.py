import networkx as nx
from gensim.models import Word2Vec
import initialization
import random
from statistics import mean
import cPickle
import csv
import evonrl 
import argparse
from elasticsearch import Elasticsearch
from elasticsearch import helpers

def parse_args():
	'''
	Parses the  arguments.
	'''
	parser = argparse.ArgumentParser(description="Run EvoNRL.")

	parser.add_argument('--input', nargs='?', default='edges.csv',
	                    help='Input graph path')

	parser.add_argument('--edges', nargs='?', default='edges_evolution.csv',
	                    help='Input edges path')

	parser.add_argument('--vecinput', nargs='?', default='node2vec_sameinits/inits/inits_bc.vec',
	                    help='Initialization of the Embeddings input path')

	parser.add_argument('--output', nargs='?', default='sample',
	                    help='Embeddings path')

	parser.add_argument('--walks-output', nargs='?', default='walks.txt',
	                    help='updated walks output path')

	parser.add_argument('--vecoutput', nargs='?', default='node2vec_sameinits/inits/inits_bc.vec',
	                    help='Initialization of the Embeddings output path')

	parser.add_argument('--walkfile', nargs='?', default='evonrlsw_walks_v0.pkl',
	                    help='Initial walk file path if available')

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

	parser.add_argument('--csv', type=bool, default=False,
	                    help='graph input is csv format')

	parser.add_argument('--totalsteps', type=int, default=10,
	                    help='number of totaledges added.')
	return parser.parse_args()

# function to read the graphs
def graphs():
	'''
	This function reads the edgelist and creates the graph
	return: networkx graph object
	'''
	if args.csv:
		g = nx.read_edgelist(args.input, delimiter=',', nodetype=int)
	else:
		g = nx.read_edgelist(args.input, nodetype=int)
	for edge in g.edges():
		g[edge[0]][edge[1]]['weight'] = 1
	return g

# divides a list into smaller lists
def chunks(l, n):
	'''
	This function starts from the begining and wraps every n consequetive\\
	elements of l into a new list
	'''
    for i in range(0, len(l), n):
        yield l[i:i + n]

def create_edgelist():
	edges_evolve = []
	with open(args.edges, 'r') as f:
		for line in f:
			line = line.rstrip('\n')
			edges_evolve.append(((int(line.split(',')[0]), int(line.split(',')[1])), line.split(',')[2]))
	return edges_evolve


def edgeloops(g, walks, es):
	edges_evolve = create_edgelist()
	edges_evolve = chunks(edges_evolve, args.totalsteps) 
	steps = 0
	for chunk in edges_evolve:
		lr, walks, keys = evonrl.main(g, walks, num_walks= args.num_walks, edges=chunk, es=es, wl=args.walk_length, ind=args.indexx, inputvec=args.vecinput, output=args.output + str(steps))
		for edge in chunk:
			if edge[1] == '1':
				g.add_edge(*edge[0])
			else:
				g.remove_edge(*edge[0])
		steps = steps + 1
 

def main(args):
	g = graphs()	
	walks, es, vocab, keys = initialization.main(g, args.indexx , args.num_walks, args.walk_length, args.vecinput, args.output, args.dimensions, args.window_size, args.workers, args.iter, args.simulatewalks, args.walkfile)
	walks = [map(str, walk) for walk in walks]
	edgeloops(g, walks, es)

if __name__ == "__main__":
	args = parse_args()
	main(args)
