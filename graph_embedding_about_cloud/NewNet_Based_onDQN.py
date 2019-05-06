# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 23:53:23 2019

@author: janae
"""
import random
import numpy as np
import tensorflow as tf
from collections import deque 
# Hyper Parameters:
FRAME_PER_ACTION = 1
GAMMA = 0.8 # decay rate of past observations
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
S2V_iteration_times=1

class S2V_NET:

	def __init__(self,actions):
		# init replay memory
		self.replayMemory = deque()
		# init some parameters
		self.timeStep = 0
		self.epsilon = INITIAL_EPSILON
		self.actions = actions
		# init Q network
		self.create_S2V_QNetwork()
		
		
        
	def get_key (dict, value):#find the number of jobs or services
	    return [k for k, v in dict.items() if v == value]        

	def create_S2V_QNetwork(self):
		# s2v parameters
		self.node_label_para = tf.variable(np.random.randint(100,size=(node_dim, feature_embedding_dim))/10003)#node_label:theta1

		self.node_feature = tf.variable(np.random.randint(100,size=(feature_embedding_dim, feature_embedding_dim))/10003)#node_feature:theta2	
    
		self.edge_weight = tf.variable(np.random.randint(100,size=(edge_dim, feature_embedding_dim))/10003)#edge_weight:theta4
		
		self.edge_aggregate = tf.variable(np.random.randint(100,size=(feature_embedding_dim, feature_embedding_dim))/10003)#edge_aggregate:theta3
		
		self.cpu_para = tf.variable(np.random.randint(100,size=(node_dim, feature_embedding_dim))/10003)#cpu_para:theta5
		
		self.M_para = tf.variable(np.random.randint(100,size=(node_dim, feature_embedding_dim))/10003)#M_para:theta6
        #update graph's U_v
		
		for i in range(0,S2V_iteration_times):
		    for node in range(0,len(NODE)):
		       label_embedding=tf.matmul(self.node_label_para,NODE[node].label)
                    
		       cpu_embedding=tf.matmul(self.cpu_para,NODE[node].cpu)
                    
		       M_embedding=tf.matmul(self.M_para,NODE[node].M)
		       # dictionary:embedding_uv  N_v:nods set except the node   dictionary
		       self.nodes_embedding=tf.constant(np.zeros((1, feature_embedding_dim)))
                    
		       self.weights_relu=tf.constant(np.zeros((edge_dim, feature_embedding_dim)))
		       f_node_i=0
                    
		       if NODE[node].type=='job': #索引与NODE的索引一致
                        for f_node in range(0,len(NODE)):
                            if NODE[f_node].type =='service':
                                N_v[f_node_i]=NODE[f_node].ID
                                f_node_i=f_node_i+1
		       else:
                        for f_node in range(0,len(NODE)):
                            if NODE[f_node].type =='job':
                                N_v[f_node_i]=NODE[f_node].ID
                                f_node_i=f_node_i+1
                        
                            
                            
                        
                    
		       for node_index in range(0,len(N_v)): 
                        embedding=self.nodes_embedding+embedding_uv[N_v[node_index]]
                        self.weights_relu=self.weights_relu+tf.nn.relu(tf.matmul(edge_weight,g[node][node_idx]))
                        
		       node_embeddind_final=tf.matmul(node_feature,self.nodes_embedding)
                    
		       weight_relu_final=tf.matmul(self.edge_aggregate,self.weights_relu)
                    
		       embedding_uv[node]=tf.nn.relu(np.add(np.add(np.add(np.add(label_embedding,node_embeddind_final),weight_relu_final),cpu_embedding),M_embedding))
            
        
        #q_value parameters
		self.Node_embedding_uv_other = tf.variable(np.random.randint(100,size=(feature_embedding_dim, feature_embedding_dim))/10003)#Node_embedding_uv_other:theta8
        
		self.Node_embedding_uv = tf.variable(np.random.randint(100,size=(feature_embedding_dim, feature_embedding_dim))/10003)#Node_embedding_uv:theta9
        
		self.Qlearning_para = tf.variable(np.random.randint(100,size=(1, feature_embedding_dim+feature_embedding_dim))/10003)#Qlearning_para:theta7
		
		#Q value
		self.Uv_embedding=tf.constant(np.zeros((feature_embedding_dim,node_dim)))
        
		action=self.getAction()
		f_node_i=0
        #代码：找到与节点v相邻的节点的集合 N_v
		if node_dict[action]=='job': #索引与NODE的索引一致
                        for f_node in range(0,len(NODE)):
                            if NODE[f_node].type =='service':
                                N_v[f_node_i]=NODE[f_node].ID
                                f_node_i=f_node_i+1
		else:
                        for f_node in range(0,len(NODE)):
                            if NODE[f_node].type =='job':
                                N_v[f_node_i]=NODE[f_node].ID
                                f_node_i=f_node_i+1
		v_embedding=tf.matmul(self.Node_embedding_uv,embedding_uv[choice_node])
		for node_dix in range(0,len(N_v)): 
                 self.Uv_embedding=np.add(self.Uv_embedding,embedding_uv[N_v[node_dix]])  
		Uv_embedding_all=tf.matmul(self.Node_embedding_uv_other,Uv_embedding)
        
		self.stateInput = tf.placeholder("float",[None])
        
		self.Qvalue=tf.matmul(self.Qlearning_para,Uv_embedding_all+v_embedding)
        
		self.actionInput = tf.placeholder("float",[None])
		self.yInput = tf.placeholder("float", [None]) 
		self.cost = tf.reduce_mean(tf.square(self.yInput - self.QValue))
		self.trainStep = tf.train.GradientDescentOptimizer(0.5).minimize(self.cost)

		# saving and loading networks
		self.saver = tf.train.Saver()
		self.session = tf.InteractiveSession()
		self.session.run(tf.initialize_all_variables())
		checkpoint = tf.train.get_checkpoint_state("saved_networks")
		if checkpoint and checkpoint.model_checkpoint_path:
				self.saver.restore(self.session, checkpoint.model_checkpoint_path)
				print ("Successfully loaded:", checkpoint.model_checkpoint_path)
		else:
				print ("Could not find old network weights")

	def trainQNetwork(self):
		# Step 1: obtain random minibatch from replay memory
		minibatch = random.sample(self.replayMemory,BATCH_SIZE)
		state_batch = [data[0] for data in minibatch]
		action_batch = [data[1] for data in minibatch]
		reward_batch = [data[2] for data in minibatch]
		nextState_batch = [data[3] for data in minibatch]

		# Step 2: calculate y 
		y_batch = []
		QValue_batch = self.QValue.eval(feed_dict={self.stateInput:nextState_batch})
		for i in range(0,BATCH_SIZE):
			#判断是否是最终状态  terminal = self.isTerminal()
			if terminal: 最终状态如何判断
				y_batch.append(reward_batch[i])
			else:
				y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

		self.trainStep.run(feed_dict={
			self.yInput : y_batch,
			self.actionInput : action_batch,
			self.stateInput : state_batch
			})

		# save network every 100000 iteration
		if self.timeStep % 10000 == 0:
			self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step = self.timeStep)

		
	def setPerception(self,nextObservation,action,reward,terminal):
		#newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
		newState = np.append(self.currentState[:,:,1:],nextObservation,axis = 2)
		self.replayMemory.append((self.currentState,action,reward,newState,terminal))
		if len(self.replayMemory) > REPLAY_MEMORY:
			self.replayMemory.popleft()
		if self.timeStep > OBSERVE:
			# Train the network
			self.trainQNetwork()

		self.currentState = newState
		self.timeStep += 1

	def getAction(self):
		#有待商榷,能选择的action在此步骤上就应该有：QValue = self.QValue.eval(feed_dict= {self.stateInput:[self.currentState]})[0]
        currentstate_index=fingIndex(NODE,self.currentState)
        action,action_type=choice_avaliable_action(NODE,currentstate_index) #其中action存储的是节点的ID
        #判断service的action组合是否为空
        if len(action)==0 & action_type=='service'
            #找到job中最小的   从存储的状态序列中找，该状态序列存储的是节点的ID号  假设我们已经有该状态序列
            for i in range(0,len(NODE)):
                if NODE[i].label==1 & NODE[i].type=='job':
                    if NODE[i].cpu>=NODE[currentstate_index].cpu & NODE[i].M>=NODE[currentstate_index].M：
                        service_ID=ServiceID_fromStateList(StateList,self.currentState)
                        ServiceIDlist[k]=service_ID
                        k=k+1
            if random.random() <= self.epsilon:
                action_index = random.randrange(0,len(ServiceIDlist))
                choice_action=ServiceIDlist[action_index]
            else:
                min_time=NODE[fingIndex(NODE,ServiceIDlist[0])].time
                min_ID=NODE[fingIndex(NODE,ServiceIDlist[0])].ID
                for k in range(0,len(ServiceIDlist)):
                    if min_time>NODE[fingIndex(NODE,ServiceIDlist[k])].time:
                        min_time=NODE[fingIndex(NODE,ServiceIDlist[k])].time
                        min_ID=NODE[fingIndex(NODE,ServiceIDlist[k])].ID
                choice_action=min_ID
                        
		else:
            
            action_index = 0
		
            if random.random() <= self.epsilon:
				action_index = random.randrange(0,len(action))
				choice_action = action[action_index]
            else:
				action_index = np.argmax(QValue)
				choice_action = action[action_index]
		

		# change episilon
		if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
			self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/EXPLORE

		return choice_action

	def setInitState(self,observation):
		self.currentState = np.stack((observation, observation, observation, observation), axis = 2)

	def fingIndex(NODE,ID):
        for i in range(0,len(NODE)):
            if NODE[i].ID==ID:
                return i
    
    def choice_avaliable_action(NODE,currentstate_index):
        k=0
		if NODE[currentstate_index].type=='job':#在当前状态下可以选择的action的集合，可选择的action，除去label为1的节点以及判断其是否达到终止状态，即job的label是否都为1
            for i in range(0,len(NODE)):
                if NODE[i].type=='service' & NODE[i].label==0:
                    action[k]=NODE[i].ID
                    k=k+1
            action_type='service'
        else:
            for i in range(0,len(NODE)):
                if NODE[i].type=='job' & NODE[i].label==0:
                    action[k]=NODE[i].ID
                    k=k+1
            action_type='job'
        return action,action_type
                
            
	def ServiceID_fromStateList(StateList,currentstate):
        for i in range(0,len(StateList)):
            if StateList[i]==currentstate:
               return StateList[i+1]
                
        
	def conv2d(self,x, W, stride):
		return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

	def max_pool_2x2(self,x):
		return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
		
