# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 15:28:52 2019

@author: janae
"""
import random
import numpy as np
import tensorflow as tf

# Hyper Parameters:
FRAME_PER_ACTION = 1
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100. # timesteps to observe before training
EXPLORE = 200000. 
FINAL_EPSILON = 0.05 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
STEP_EPSILON=10000.
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH_SIZE = 32 # size of minibatch
node_dim=1
feature_embedding_dim=64
edge_dim=1

def build_s2v(g,node_dict,job,service,node_abel):
    node_label_para=np.random.randn(node_dim, feature_embedding_dim) #node_label:theta1
    node_feature=np.random.randn(feature_embedding_dim, feature_embedding_dim)#node_feature:theta2
    edge_weight=np.random.randn(edge_dim, feature_embedding_dim)#edge_weight:theta4
    edge_aggregate=np.random.randn(feature_embedding_dim, feature_embedding_dim)#edge_aggregate:theta3
    cpu_para=np.random.randn(node_dim, feature_embedding_dim) #cpu_para:theta5
    M_para=np.random.randn(node_dim, feature_embedding_dim) #M_para:theta6
    
    label_embedding=np.dot(node_label_para,node_abel)
    cpu_embedding=np.dot(cpu_para,choice_cpu)
    M_embedding=np.dot(M_para,choice_M)
    # dictionary:embedding_uv  N_v:node set except choice_node   dictionary
    nodes_embedding=np.zeros((node_dim, feature_embedding_dim))
    weights_relu=np.zeros((edge_dim, feature_embedding_dim))
    for node_dix in N_v: 
        nodes_embedding=nodes_embedding+embedding_uv[node_dix]
        weights_relu=weights_relu+tf.nn.relu(np.dot(edge_weight,g[choice_node][node_idx]))
    node_embeddind_final=np.dot(node_feature,nodes_embedding)
    weight_relu_final=np.dot(edge_aggregate,weights_relu)
    embedding_uv[choice_node]=tf.nn.relu(np.add(np.add(np.add(np.add(label_embedding,node_embeddind_final),weight_relu_final),cpu_embedding),M_embedding))
    
    return embedding_uv[choice_node]

def q_learning():
    s
    
