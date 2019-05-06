# -*- coding: utf-8 -*-
"""
Created on Sun May  5 15:30:55 2019

@author: janae
"""
import numpy as np
print(np.random.randint(100,size=(64, 64))/10003)



original parameters:
        self.node_label_para = tf.variable(np.random.randn(node_dim, feature_embedding_dim))#node_label:theta1

		self.node_feature = tf.variable(np.random.randn(feature_embedding_dim, feature_embedding_dim))#node_feature:theta2	
    
		self.edge_weight = tf.variable(np.random.randn(edge_dim, feature_embedding_dim))#edge_weight:theta4
		
		self.edge_aggregate = tf.variable(np.random.randn(feature_embedding_dim, feature_embedding_dim))#edge_aggregate:theta3
		
		self.cpu_para = tf.variable(np.random.randn(node_dim, feature_embedding_dim))#cpu_para:theta5
		
		self.M_para = tf.variable(np.random.randn(node_dim, feature_embedding_dim))#M_para:theta6
        
        
        
        
        
        
        #q_value parameters
		self.Node_embedding_uv_other = tf.variable(np.random.randn(feature_embedding_dim, feature_embedding_dim))#Node_embedding_uv_other:theta8
        
		self.Node_embedding_uv = tf.variable(np.random.randn(feature_embedding_dim, feature_embedding_dim))#Node_embedding_uv:theta9
        
		self.Qlearning_para = tf.variable(np.random.randn(1, feature_embedding_dim+feature_embedding_dim))#Qlearning_para:theta7
		