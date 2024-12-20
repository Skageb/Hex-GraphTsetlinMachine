#!/usr/bin/env python
# coding: utf-8

# # Load Dataset

# In[ ]:


import argparse
import pandas as pd
import numpy as np
from time import monotonic

parser = argparse.ArgumentParser(description="Set board size for Hex dataset.")
parser.add_argument("--board_size", type=int, default=7, help="Size of the Hex board (default: 7)")
parser.add_argument("--run_number", type=int, default=-1,  help="For average accuracy")
parser.add_argument("--n_turns", type=int, default=0,  help="For average accuracy")

args = parser.parse_args()

board_size = args.board_size
run_number = args.run_number

n_turns_before_win = args.n_turns
n_games = 10000

path = f'hex_datasets/hex_{board_size}x{board_size}_{n_games}_games_{n_turns_before_win}_turns_before_win.csv'

df = pd.read_csv(path)

train_split = 0.8

train_end_index = int(len(df)*train_split)

print(train_end_index)

df_train = df[0:train_end_index]
df_test = df[train_end_index:]
print(len(df_train), len(df_test))


# In[14]:


from GraphTsetlinMachine.graphs import Graphs
start_graph_init = monotonic()
HYPERVECTOR_SIZE = 120
HYPERVECTOR_BITS = 2

x_positon_symbols = [f'x:{x}' for x in range(board_size + 2)]
y_positon_symbols = [f'y:{y}' for y in range(board_size + 2)]

graph_node_args = {
    'symbols' : ['PlayerX', 'PlayerXEdge', 'Empty','PlayerO', 'PlayerOEdge'] + x_positon_symbols + y_positon_symbols,
    'hypervector_size': HYPERVECTOR_SIZE,
    'hypervector_bits': HYPERVECTOR_BITS
}


assert board_size == int(np.sqrt(len(df_train.iloc[0, :-1])))

num_nodes = board_size*board_size + (2*(board_size+1)) + (2 * board_size)     #Adds edges as colored nodes

print(num_nodes)


# ## Get all Nodes for the board dimension and add initialize graphs

# In[15]:


all_node_names = [f'{i}:{j}' for i in range(board_size+2) for j in range(board_size+2)]
print(all_node_names)

all_node_names = all_node_names[1:-1]     #Get all nodes including edges represented as nodes
print(all_node_names)


graphs_train = Graphs(
    train_end_index,
    **graph_node_args
)


graphs_test = Graphs(
    len(df_test),
    **graph_node_args,
    init_with=graphs_train
)

#Initalize graphs
for graph_id in range(len(df_train)):
    graphs_train.set_number_of_graph_nodes(graph_id, number_of_graph_nodes=np.uint32(len(all_node_names)))

for graph_id in range(len(df_test)):
    graphs_test.set_number_of_graph_nodes(graph_id, number_of_graph_nodes=np.uint32(len(all_node_names)))

    


# # Initalize edges

# In[16]:


graphs_train.prepare_node_configuration()
graphs_test.prepare_node_configuration()
from copy import copy

# Add corner edge nodes
remianing_nodes = copy(all_node_names)
corner_nodes = ['0:1', '1:0', f'{board_size}:{board_size+1}', f'{board_size+1}:{board_size}', f'{0}:{board_size+1}', f'{board_size+1}:{0}']
print('Corner Nodes:', corner_nodes)
for node_name in corner_nodes:
    remianing_nodes.remove(node_name)

#Add remaining edge nodes
remaining_edge_nodes = [node for node in remianing_nodes if ('0' in node.split(':') or f'{board_size+1}' in node.split(':'))]
print('Remaining Edge Nodes:', remaining_edge_nodes)
for node_name in remaining_edge_nodes:
    remianing_nodes.remove(node_name)

print('Number of remaining Board Nodes:', len(remianing_nodes))



#Initalise nodes in train graph
for graph_id in range(len(df_train)):
    for node_name in corner_nodes:
        graphs_train.add_graph_node(graph_id, node_name=node_name, number_of_graph_node_edges=1)
    
    for node_name in remaining_edge_nodes:
        graphs_train.add_graph_node(graph_id, node_name=node_name, number_of_graph_node_edges=2)
    
    for node_name in remianing_nodes:
        graphs_train.add_graph_node(graph_id, node_name=node_name, number_of_graph_node_edges=6)
    

for graph_id in range(len(df_test)):
    for node_name in corner_nodes:
        graphs_test.add_graph_node(graph_id, node_name=node_name, number_of_graph_node_edges=1)
    
    for node_name in remaining_edge_nodes:
        graphs_test.add_graph_node(graph_id, node_name=node_name, number_of_graph_node_edges=2)
    
    for node_name in remianing_nodes:
        graphs_test.add_graph_node(graph_id, node_name=node_name, number_of_graph_node_edges=6)
    
    

    


# # Add Edges

# In[17]:


graphs_train.prepare_edge_configuration()
graphs_test.prepare_edge_configuration()

directions = {
    (1, -1) : 'UpRight',
    (0, -1) : 'UpLeft',
    (1, 0) : 'Right',
    (-1, 0) : 'Left',
    (0, 1) : 'DownRight',
    (-1, 1) : 'DownLeft'
}


def add_offset_to_node(node_name, offset):
    #print(node_name, offset)
    i, j = map(int, node_name.split(':'))
    i += offset[0]
    j += offset[1]
    return ':'.join([str(i), str(j)])

all_offsets = directions.keys()
    


#Add remaining edge nodes
edges = []

#Add edges for all edge nodes
for node_name in all_node_names:
    for offset in all_offsets:
        neighbor = add_offset_to_node(node_name, offset)
        #print(neighbor)
        if node_name in corner_nodes or node_name in remaining_edge_nodes:      #For edge nodes, only add game tiles as edges.
            if neighbor in remianing_nodes:
                edges.append({
                    'from': node_name,
                    'to': neighbor,
                    'edge_type': directions[offset]
                })
        else:
            edges.append({
                    'from': node_name,
                    'to': neighbor,
                    'edge_type': directions[offset]
                })





for graph_id in range(len(df_train)):
    for edge in edges:
        graphs_train.add_graph_node_edge(graph_id, edge['from'], edge['to'], edge['edge_type'])

for graph_id in range(len(df_test)):
    for edge in edges:
        graphs_test.add_graph_node_edge(graph_id, edge['from'], edge['to'], edge['edge_type'])


        


# # Add Graph Node Property and Extract winning player

# In[18]:


Y_train = np.empty(len(df_train), dtype=np.uint32)

for graph_id, row in df_train.iterrows():
    for node_name in all_node_names:
        if node_name.split(":")[1] in ['0', f'{board_size+1}']:
            graphs_train.add_graph_node_property(graph_id, node_name=node_name, symbol='PlayerXEdge')
        elif node_name.split(":")[0] in ['0', f'{board_size+1}']:
            graphs_train.add_graph_node_property(graph_id, node_name=node_name, symbol='PlayerOEdge')
        else:
            assert node_name in remianing_nodes
            cell_value = row[f'cell{add_offset_to_node(node_name, (-1, -1)).replace(':', '_')}']
            if cell_value == -1:
                symbol = 'PlayerO'
            elif cell_value == 0:
                symbol = 'Empty'
            elif cell_value == 1:
                symbol = 'PlayerX'

            graphs_train.add_graph_node_property(graph_id, node_name=node_name, symbol=symbol)
        graphs_train.add_graph_node_property(graph_id, node_name=node_name, symbol=f'x:{node_name.split(':')[0]}')   #add x position symbol
        graphs_train.add_graph_node_property(graph_id, node_name=node_name, symbol=f'y:{node_name.split(':')[1]}')   #add y position symbol
    # Get winning player
    Y_train[graph_id] = row['winner'] if row['winner'] == 1 else 2

graphs_train.encode()


Y_test = np.empty(len(df_test), dtype=np.uint32)

for graph_id, row in df_test.iterrows():
    graph_id = graph_id - len(df_train)
    for node_name in all_node_names:
        if node_name.split(":")[1] in ['0', f'{board_size+1}']:
            graphs_test.add_graph_node_property(graph_id, node_name=node_name, symbol='PlayerXEdge')
        elif node_name.split(":")[0] in ['0', f'{board_size+1}']:
            graphs_test.add_graph_node_property(graph_id, node_name=node_name, symbol='PlayerOEdge')
        else:
            assert node_name in remianing_nodes
            cell_value = row[f'cell{add_offset_to_node(node_name, (-1, -1)).replace(':', '_')}']
            if cell_value == -1:
                symbol = 'PlayerO'
            elif cell_value == 0:
                symbol = 'Empty'
            elif cell_value == 1:
                symbol = 'PlayerX'
            graphs_test.add_graph_node_property(graph_id, node_name=node_name, symbol=symbol)
        graphs_test.add_graph_node_property(graph_id, node_name=node_name, symbol=f'x:{node_name.split(':')[0]}')   #add x position symbol
        graphs_test.add_graph_node_property(graph_id, node_name=node_name, symbol=f'y:{node_name.split(':')[1]}')   #add y position symbol
    # Get winning player
    Y_test[graph_id] = row['winner'] if row['winner'] == 1 else 2

graphs_test.encode()

end_graph_init = monotonic()
graph_init_time = end_graph_init - start_graph_init
print(f'Graphs initalised in time: {graph_init_time}s')

# In[19]:


##Logging function
import json
import os
def log_if_best_result(accuracy_eval, **kwargs):
    existing_result = False
    log = False
    
        
    
    if run_number != -1:    #Default value
        result_folder = os.path.join('results/', f'{board_size}x{board_size}_board_{n_turns_before_win}_turns_before_win')
        file_name = f'run_{run_number}.json'
            
    else:
        result_folder = 'results/'
        file_name = f'{board_size}x{board_size}_board_{n_turns_before_win}_turns_before_win.json'
        
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    path = os.path.join(result_folder, file_name)
    
    log_object = {'eval_accuracy' : accuracy_eval, 'n_train_games': n_games, **kwargs}
    if file_name not in os.listdir(result_folder):
        log = True
    else:
        existing_result = True
        with open(path, 'r') as f:
            json_object = json.load(f)
            best_accuracy = json_object['eval_accuracy']
            if accuracy_eval > best_accuracy:
                log = True
                
            elif accuracy_eval == 100:
                if log_object['TM_args']['number_of_clauses'] < json_object['TM_args']['number_of_clauses']:
                    log = True
    if log:
        print(f'New best result for board size {board_size}, {n_turns_before_win} turns before win\nAccuracy : {accuracy_eval} with {log_object['TM_args']['number_of_clauses']} clauses.')
        if existing_result:
            print(f'Previouis best accuracy was: {best_accuracy} with {json_object['TM_args']['number_of_clauses']} clauses.')
        print('Logging paramters')
        with open(path, 'w') as f:
            json.dump(log_object, f, indent=4)


# # Smoke test

# In[20]:


from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from time import time





# In[21]:


from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from time import time



#TM Hyper paremeters            Include defeault values in NoisyXOR
TM_args = {
    'number_of_clauses' : 7500,           #Default    10
    'T' : 3750,                          #Default   100
    's' : 2.0,                          #Default   1.0
    'number_of_state_bits' : 32,        #Default     2
    'depth' : 1,                        #Default     8
    'message_size' : 256,              #Default   256
    'message_bits' : 2,                 #Default     2
    'max_included_literals' : 100,       #Default     4
    'double_hashing': True              #Default False
}

epochs_elapsed, train_time_elapsed = 0, 0

train_accuracy_across_epochs, test_accuarcy_across_epochs = [], []


tm = MultiClassGraphTsetlinMachine(
    **TM_args,
    grid = (1, 1, 1),
    block = (256, 1, 1)
)
values, counts = np.unique_counts(Y_train)
print(f'Class distribution: Class {values[0]}: {counts[0]}, Class {values[1]}: {counts[1]}')
print(f'Dummy accuracy = {max(counts)/sum(counts)}')


# In[ ]:


assert len(graphs_train.number_of_graph_nodes) == len(Y_train)
#Train and Eval
EPOCHS = 200

#print(f'Prior to training: Accuracy_train: {100*(tm.predict(graphs_train) == Y_train).mean()},  Accuracy_test: {100*(tm.predict(graphs_test) == Y_test).mean()}')

for i in range(EPOCHS):
    epochs_elapsed += 1
    start_training = monotonic()
    tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
    stop_training = monotonic()
    epoch_train_time = stop_training-start_training
    train_time_elapsed += epoch_train_time      


    start_testing = monotonic()
    result_test = 100*(tm.predict(graphs_test) == Y_test).mean()
    stop_testing = monotonic()

    result_train = 100*(tm.predict(graphs_train) == Y_train).mean()

    from sklearn.metrics import f1_score
    f1_test = f1_score(tm.predict(graphs_test), Y_test)

    train_accuracy_across_epochs.append(result_train)
    test_accuarcy_across_epochs.append(result_test)

    print("Epoch: %d, Accuracy_train: %.2f,  Accuracy_test: %.2f, Accumulated Training Time: %.2f, Testing time: %.2f" % (epochs_elapsed, result_train, result_test, train_time_elapsed, stop_testing-start_testing))

    log_if_best_result(result_test,
                       f1_test = f1_test, 
                       train_accuracy=result_train, 
                       EPOCHS=epochs_elapsed, 
                       hypervector_size=HYPERVECTOR_SIZE, 
                       hypervector_bits=HYPERVECTOR_BITS,
                       train_time=train_time_elapsed,
                       graph_initalization_time = graph_init_time,
                       TM_args=TM_args,
                       train_accuracy_across_epochs=train_accuracy_across_epochs,
                       test_accuarcy_across_epochs=test_accuarcy_across_epochs)

    if result_test == 100:
        print('100% Accuracy achieved on testset!')
        break
         
         



# # 2x2 Board 100 % Accuracy

# In[ ]:


#weights = tm.get_state()[1].reshape(2, -1)
#for i in range(tm.number_of_clauses):
#        print("Clause #%d W:(%d %d)" % (i, weights[0,i], weights[1,i]), end=' ')
#        l = []
#        for k in range(HYPERVECTOR_SIZE * 2):
#            if tm.ta_action(0, i, k):
#                if k < HYPERVECTOR_SIZE:
#                    l.append("x%d" % (k))
#                else:
#                    l.append("NOT x%d" % (k - HYPERVECTOR_SIZE))

        # for k in range(args.message_size * 2):
        #     if tm.ta_action(1, i, k):
        #         if k < args.message_size:
        #             l.append("c%d" % (k))
        #         else:
        #             l.append("NOT c%d" % (k - args.message_size))

#        print(" AND ".join(l))

#print(graphs_train.hypervectors)
#print(tm.hypervectors)
#print(graphs_train.edge_type_id)


# In[ ]:


number_of_nodes = len(all_node_names)
aproc_number_of_edges = number_of_nodes*6
recommended_clauses = 5*aproc_number_of_edges*number_of_nodes
#print(recommended_clauses)


# In[ ]:




