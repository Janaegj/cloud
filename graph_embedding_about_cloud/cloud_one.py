# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 21:05:49 2019

@author: janae
"""

import numpy as np
import networkx as nx
from networkx.algorithms import bipartite


import random
from random import choice

class Node_Information:
    def __init__(self):
        self.ID ='' 
        self.time =float( 0 )
        self.cpu =float( 0 )
        self.M = float( 0 )	


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
	
	
	pathtofile_job='D:/CLOUD/job(no level).txt'
	pathtofile_service='D:/CLOUD/service.txt'
	i_job=0
	i_service=0
	
	
	with open(pathtofile_job) as f_job:
            content_job = f_job.readlines()
	content_job = [x_job.strip() for x_job in content_job]
	content_job = content_job[1:]
	    
	j = 0
	for i in content_job:
            j=j+1  
	
	job=[Node_Information() for i in range(j)]
	
	
    
	for line_job in content_job:
			
			entries_job = line_job.split()
			job[i_job].ID = entries_job[0]
			job[i_job].time = entries_job[1]
			job[i_job].cpu = entries_job[2]
			job[i_job].M = entries_job[3]
			i_job=i_job+1
	
	with open(pathtofile_service) as f_service:
		content_service = f_service.readlines()
	content_service = [x_service.strip() for x_service in content_service]
	content_service = content_service[1:]
	
    
	j=0
	for i in content_service:
            j=j+1  
	
	service=[Node_Information() for i in range(j)]
	
			
	for line_service in content_service:
			
			entries_service = line_service.split()
			service[i_service].ID = entries_service[0]
			service[i_service].cpu = entries_service[1]
			service[i_service].M = entries_service[2]
			i_service=i_service+1
			
			
	with open(pathtofile) as f:
		content = f.readlines()
	content = [x.strip() for x in content]
	content = content[1:]
	
	
	for line in content:
		entries = line.split()  #entries下标从0开始
		job_str = entries[0] 
		service_str = entries[1]
		
		if job_str not in node_dict:
			node_dict[job_str] = len(node_dict)
			g.add_node(node_dict[job_str])
		if service_str not in node_dict:
			node_dict[service_str] = len(node_dict)
			g.add_node(node_dict[service_str])

		job_idx = node_dict[job_str]
		service_idx = node_dict[service_str]

		
		
		
        
		for i in range(j):
			if job[i].ID==job_str:
			     weight=job[i].time
		g.add_edge(job_idx,service_idx,weight=float(weight) ,count= 1)

		times.append(float(weight))

	for edge in g.edges(data=True):
		job_idx = edge[0]
		service_idx = edge[1]
		w = edge[2]['weight']
		
		g[job_idx][service_idx]['weight'] = w

	return g, node_dict,job,service

def get_key (dict, value):
    return [k for k, v in dict.items() if v == value]

def get_cm(Service, ID,number_of_service):
    c=0
    m=0
    for i in range(len(number_of_service)):
       if Service[i].ID==ID[0]:
           c=Service[i].cpu
           m=Service[i].M
    return c,m
    
    

def get_cloud_graph(ig,node_dict,job,service):
	g = ig.copy()
	Job = job.copy()
	Service = service.copy()
	s_cpu=[]
	s_m=[]
	s=[]
    
	job_nodes_idx_set, service_nodes_idx_set = bipartite.sets(g_undirected) #得到的是集合，要将集合转为列表
	job_nodes_idx=list(job_nodes_idx_set)
	service_nodes_idx=list(service_nodes_idx_set)
	job_node_idx=choice(job_nodes_idx)
	job_node=get_key(node_dict,job_node_idx) #得到job的ID，便于找寻对应ID的CPU和内存
	job_cpu,job_m=get_cm(Job,job_node,job_nodes_idx) 
	print(type(job_cpu))
	service_cpu = range(len(service_nodes_idx))
	
    #得到service的CPU和内存,返回数值是对应service坐标的内存和CPU
	
	for i in service_nodes_idx:
            service_node=get_key(node_dict,i)
            service_cpu,service_m=get_cm(Service,service_node,service_nodes_idx)
            s_cpu.append (float(service_cpu))
            
            s_m.append (float(service_m))
            s.append(i)
            
            
          
   #根据CPU和内存进行排序 还要选择要哪个service 
	s_cpui=s
	s_mi=s
	min_cpu=0
	min_cpui=0
	min_m=0
	min_mi=0
	for k in range(len(s_cpu)):
	   for k_i in range(k,len(s_cpu)):
	     if s_cpu[k_i]<s_cpu[k]:
               min_cpu=s_cpu[k_i]
               s_cpu[k_i]=s_cpu[k]
               s_cpu[k]=min_cpu
               min_cpui=s_cpui[k_i]
               s_cpui[k_i]=s_cpui[k]
               s_cpui[k]= min_cpui
            
	for k in range(len(s_m)):
	   for k_i in range(k,len(s_cpu)):
	     if s_m[k_i]<s_m[k]:
               min_m=s_m[k_i]
               s_m[k_i]=s_m[k]
               s_m[k]=min_m
               min_mi=s_mi[k_i]
               s_mi[k_i]=s_mi[k]
               s_mi[k]= min_mi
               
   #找特定job的    CPU和内存
	
	idx=[]
	for k in range(len(s_cpui)):
	   if float(job_cpu)<=s_cpu[k] and float(job_m)<= s_m[k]:
	     if s_mi[k]==s_cpui[k]:
                idx.append(s_cpui[k])
                
	   else:
            #把两点之间边的权重改为无限大  一般设为9999
            g[job_node_idx][s_mi[k]]['weight'] = 9999
            g[job_node_idx][s_cpui[k]]['weight'] = 9999
            
	service_node_idx_choice=choice(idx)
	print(idx)
	print('~~~~~~~~~~~')
	print(service_node_idx_choice)
   
   #找到对应该服务器的ID号，然后减去运行该job后还能用的内存和CPU
	service_node=get_key(node_dict,service_node_idx_choice)
	print(service_node[0])
	print(range(len(service_nodes_idx)))
	
	for i in range(len(service_nodes_idx)):
	  if Service[i].ID==service_node[0]:
           print(Service[i].cpu)
           Service[i].cpu=float(Service[i].cpu)-float(job_cpu)
           print(Service[i].cpu)
           Service[i].M=float(Service[i].M)-float(job_m)

	return g,Job,Service

def visualize(g,pdfname='graph.pdf'):
	import matplotlib.pyplot as plt
	pos=nx.spring_layout(g,iterations=100) # positions for all nodes
	nx.draw_networkx_nodes(g,pos,node_size=1)
	nx.draw_networkx_edges(g,pos)
	plt.axis('off')
	plt.savefig(pdfname,bbox_inches="tight")












if __name__ == '__main__':
	# build full graphs of both types
	print("Building undirected graph ...")
	g_undirected, node_dict,job,service = build_full_graph('D:/CLOUD/allocation.txt','undirected')
	g, j,s=get_cloud_graph(g_undirected,node_dict,job,service)
	job_nodes_idx_set, service_nodes_idx_set = bipartite.sets(g)
	print(g_undirected.edges)
	print( 'number of edges:', g_undirected.size())
	
	
    
	#print(g_undirected)
	#print( 'number of edges:', g_undirected.size())
	#print(g_undirected.edges)
	#print(nx.is_bipartite(g_undirected))

	#visualize(g_directed)

	#print(nx.number_of_nodes(g_undirected))
	#print(nx.number_of_edges(g_undirected))

	#print(nx.number_of_nodes(g_directed))
	#print(nx.number_of_edges(g_directed))

