# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
import networkx as nx
import random




    

def build_full_graph(pathtofile, graphtype):
	node_dict = {}

	if graphtype == 'undirected':
		g = nx.Graph()
	elif graphtype == 'directed':
		g = nx.DiGraph()
	else:
		print('Unrecognized graph type .. aborting!')
		return -1

	times = []
	with open(pathtofile) as f:
		content = f.readlines()
	content = [x.strip() for x in content]
	content = content[1:]

	for line in content:
		entries = line.split()  #entries下标从0开始
		src_str = entries[1] 
		dst_str = entries[2]

		if src_str not in node_dict:
			node_dict[src_str] = len(node_dict)
			g.add_node(node_dict[src_str])
		if dst_str not in node_dict:
			node_dict[dst_str] = len(node_dict)
			g.add_node(node_dict[dst_str])

		src_idx = node_dict[src_str]
		dst_idx = node_dict[dst_str]

		w = 0
		c = 0
		if g.has_edge(src_idx,dst_idx):
			w = g[src_idx][dst_idx]['weight']
			c = g[src_idx][dst_idx]['count']

		g.add_edge(src_idx,dst_idx,weight=w + 1.0/float(entries[-1]),count=c + 1)

		times.append(float(entries[-1]))

	for edge in g.edges(data=True):
		src_idx = edge[0]
		dst_idx = edge[1]
		w = edge[2]['weight']
		c = edge[2]['count']
		g[src_idx][dst_idx]['weight'] = w/c

	return g, node_dict



def visualize(g,pdfname='graph.pdf'):
	import matplotlib.pyplot as plt
	pos=nx.spring_layout(g,iterations=100) # positions for all nodes
	nx.draw_networkx_nodes(g,pos,node_size=1)
	nx.draw_networkx_edges(g,pos)
	plt.axis('off')
	plt.savefig(pdfname,bbox_inches="tight")

def visualize_bipartite(g,pdfname='graph_scp.pdf'):
	import matplotlib.pyplot as plt
	X, Y = nx.bipartite.sets(g)
	pos = dict()
	pos.update( (n, (1, i)) for i, n in enumerate(X) ) # put nodes from X at x=1
	pos.update( (n, (2, i)) for i, n in enumerate(Y) ) # put nodes from Y at x=2
	nx.draw_networkx_nodes(g,pos,node_size=1)
	nx.draw_networkx_edges(g,pos)
	plt.axis('off')
	plt.savefig(pdfname,bbox_inches="tight")

if __name__ == '__main__':
	# build full graphs of both types
	print("Building undirected graph ...")
	g_undirected, node_dict = build_full_graph('G:\graph_comb_opt\code\memetracker\InfoNet5000Q1000NEXP.txt','undirected')
    
	print("Building directed graph ...")
	g_directed, node_dict = build_full_graph('G:\graph_comb_opt\code\memetracker\InfoNet5000Q1000NEXP.txt','directed')

	#print(nx.number_of_nodes(g_undirected))
	#print(nx.number_of_edges(g_undirected))

	#print(nx.number_of_nodes(g_directed))
	#print(nx.number_of_edges(g_directed))
 