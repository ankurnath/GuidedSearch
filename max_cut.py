# -*- coding: utf-8 -*-


import networkx as nx
import numpy as np
import pickle
from collections import defaultdict
import time
import pandas as pd
import os
from  numba import njit
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import defaultdict
import os
import random
import networkx as nx
import time
from multiprocessing import Process
from scipy.sparse import csr_matrix,save_npz,load_npz

"""# Helper functions"""

import networkx as nx

def graph_gen(model='ER', **kwargs):
    # Extract the number of nodes 'n' from kwargs
    n = kwargs['n']

    # If the model is 'ER' (Erdős-Rényi)
    if model == 'ER':
        # Extract the probability 'p' from kwargs
        p = kwargs['p']
        # Return an Erdős-Rényi graph
        return nx.fast_gnp_random_graph(n, p)

    # If the model is 'BA' (Barabási-Albert)
    elif model == 'BA':
        # Extract the number of edges to attach from a new node 'm' from kwargs
        m = kwargs['m']
        # Return a Barabási-Albert graph
        return nx.barabasi_albert_graph(n, m)

    # If the model is 'Watts_Strogatz' (Watts-Strogatz small-world)
    elif model == 'Watts_Strogatz':
        # Extract the number of nearest neighbors 'k' and the rewiring probability 'p' from kwargs
        k = kwargs['k']
        p = kwargs['p']
        # Return a Watts-Strogatz graph
        return nx.watts_strogatz_graph(n, k, p)

    # If an unknown model is provided, raise an error
    else:
        raise NotImplementedError('Unknown model of graph')



def flatten_graph(graph):
    flat_adj_matrix = []
    flat_weight_matrix = []
    n = graph.number_of_nodes()
    start = [0 for _ in range(n)]
    end = [0 for _ in range(n)]
    adj_list_dict = nx.to_dict_of_lists(graph)

    # Assign random weights (1 or -1) to edges
    for edge in graph.edges():
#         graph.edges[edge]['weight'] = random.choice([1, -1])
        graph.edges[edge]['weight'] = 1

    for node, neighbors in adj_list_dict.items():
        start[node] = len(flat_adj_matrix)
        end[node] = start[node] + len(neighbors)
        flat_adj_matrix += neighbors

        # Build the flattened weight matrix
        flat_weight_matrix += [graph.edges[(node, neighbor)]['weight'] for neighbor in neighbors]

    return np.array(flat_adj_matrix), np.array(flat_weight_matrix), np.array(start), np.array(end)


@njit
def ls_greedy(adj_matrix, weight_matrix, start_list, end_list,size_constraint):

    number_of_queries=0


    n=len(start_list)
    merginal_gain=np.zeros(n)
    spins=np.zeros(n)


    for i in range(n):
        number_of_queries+=1
        for j,weight in zip(adj_matrix[start_list[i]:end_list[i]],
                  weight_matrix[start_list[i]:end_list[i]]):

            merginal_gain[i]+=weight*(2*spins[i]-1)*(2*spins[j]-1)




    curr_score=0
    step=0

    for _ in range(size_constraint):


        max_gain = -np.inf
        max_gain_node = -1

        for i in range(len(spins)):

            if spins[i] == 0:

                number_of_queries+=1

                if merginal_gain[i] > max_gain:

                    max_gain = merginal_gain[i]
                    max_gain_node = i



        if merginal_gain[max_gain_node]<=0:
            break


#         assert spins[max_gain_node]==0
        step+=1

        curr_score+=merginal_gain[max_gain_node]
        merginal_gain[max_gain_node]=-merginal_gain[max_gain_node]
        for u,weight in zip(adj_matrix[start_list[max_gain_node]:end_list[max_gain_node]],
                 weight_matrix[start_list[max_gain_node]:end_list[max_gain_node]]):
            merginal_gain[u]+=weight*(2*spins[u]-1)*(2-4*spins[max_gain_node])

        spins[max_gain_node] = 1-spins[max_gain_node]

    return curr_score,spins,number_of_queries

@njit
def fls_greedy(adj_matrix, weight_matrix, start_list, end_list,size_constraint,error_rate):



    n=len(start_list)
    merginal_gain=np.zeros(n)
    spins=np.zeros(n)


    number_of_queries=0

    # Calculate merginal gain for every element
    for i in range(n):
        number_of_queries+=1
        for j,weight in zip(adj_matrix[start_list[i]:end_list[i]],
                         weight_matrix[start_list[i]:end_list[i]]):

            merginal_gain[i]+=weight*(2*spins[i]-1)*(2*spins[j]-1)

    # an approximation result

    A_0=np.argmax(merginal_gain)
    k=1
    curr_score=merginal_gain[A_0]

    merginal_gain[A_0]=-merginal_gain[A_0]

    for neighbour,weight in zip(adj_matrix[start_list[A_0]:end_list[A_0]],
                     weight_matrix[start_list[A_0]:end_list[A_0]]):

        merginal_gain[neighbour]+=weight*(2*spins[neighbour]-1)*(2-4*spins[A_0])
    spins[A_0]=1-spins[A_0]

    # SWAP OR FLIP (ADD)

    continue_search=True


    while continue_search:
        best_spins=spins.copy()

        continue_search=False

        # EXCHANGE WITH DUMMY

        if k<size_constraint:
            for i in range(n):

                if spins[i] == 0:
                    number_of_queries+=1

                    if merginal_gain[i]>=(error_rate/size_constraint)*curr_score:
                        continue_search=True

                        curr_score+=merginal_gain[i]
                        merginal_gain[i]=-merginal_gain[i]


                        for u,weight in zip(adj_matrix[start_list[i]:end_list[i]],
                                         weight_matrix[start_list[i]:end_list[i]]):

                            merginal_gain[u]+=weight*(2*spins[u]-1)*(2-4*spins[i])
                        spins[i] = 1-spins[i]
                        k+=1
                        break


        #SWAP
        merginal_gain_copy=np.copy(merginal_gain)
        spins_copy=np.copy(spins)

        for e in range(n):

            if continue_search==True:
                break

            merginal_gain=np.copy(merginal_gain_copy)
            spins=np.copy(spins_copy)

            # In the solution set
            if spins[e]==1:
#                 number_of_queries+=1
                new_score=curr_score+merginal_gain[e] # (f(A-e))
                merginal_gain[e]=-merginal_gain[e]


                for u,weight in zip(adj_matrix[start_list[e]:end_list[e]],
                                     weight_matrix[start_list[e]:end_list[e]]):
                    merginal_gain[u]+=weight*(2*spins[u]-1)*(2-4*spins[e])
                spins[e] = 1-spins[e]


                for a in range(n):
                    if spins[a]==0:
                        number_of_queries+=1 # (f(A-e+a))
                        if new_score+merginal_gain[a]-curr_score>=(error_rate/size_constraint)*curr_score:

    #                         print(swap)
                            # Only if condition met then update
                            continue_search=True
                            #update
                            curr_score=new_score+merginal_gain[a]
                            merginal_gain[a]=-merginal_gain[a]
                            # for u in range(n):
                            for u,weight in zip(adj_matrix[start_list[a]:end_list[a]],
                                            weight_matrix[start_list[a]:end_list[a]]):
                                merginal_gain[u]+=weight*(2*spins[u]-1)*(2-4*spins[a])
                            spins[a] = 1-spins[a]
                            break

        # DELETE
        if continue_search is False:
            merginal_gain=np.copy(merginal_gain_copy)
            spins=np.copy(spins_copy)

            for d in range(n):
                if spins[d]==1 :
                    number_of_queries+=1
                    if merginal_gain[d]>=(error_rate/size_constraint**4)*curr_score:
                        continue_search=True
                        curr_score+=merginal_gain[d]
                        merginal_gain[d]=-merginal_gain[d]
                        for u,weight in zip(adj_matrix[start_list[d]:end_list[d]],weight_matrix[start_list[d]:end_list[d]]):

                            merginal_gain[u]+=weight*(2*spins[u]-1)*(2-4*spins[d])
                        spins[d] = 1-spins[d]
                        k-=1
                        break




    best_score=curr_score
    curr_score=0
#     print(number_of_queries)

    Z=best_spins.copy()

    spins=np.zeros(n)
    merginal_gain=np.zeros(n)
    spins=np.zeros(n)
    t=0.372
    for i in range(n):
        for j,weight in zip(adj_matrix[start_list[i]:end_list[i]],
                     weight_matrix[start_list[i]:end_list[i]]):
            merginal_gain[i]+=weight*(2*spins[i]-1)*(2*spins[j]-1)


    for i in range(1,size_constraint+1):
        arg_indices=np.argsort(-merginal_gain)

        if i<=t*size_constraint:

            indices = [index for index in arg_indices if spins[index] == 0 and Z[index]==0]
        else:
            indices = [index for index in arg_indices if spins[index] == 0]


        number_of_queries+=len(indices)
        indices=indices[:size_constraint]
        len_indices=len(indices)


        add_element=False
        for index in indices:
            if merginal_gain[index]>0:
                add_element=True
                break

        if add_element:


            rand_idx=np.random.randint(len_indices)
            rand_ele=indices[rand_idx]

            if merginal_gain[rand_ele]<=0:
                continue

            curr_score+=merginal_gain[rand_ele]

            merginal_gain[rand_ele]=-merginal_gain[rand_ele]
            for u,weight in zip(adj_matrix[start_list[rand_ele]:end_list[rand_ele]],
                            weight_matrix[start_list[rand_ele]:end_list[rand_ele]]):


                merginal_gain[u]+=weight*(2*spins[u]-1)*(2-4*spins[rand_ele])
            spins[rand_ele] = 1-spins[rand_ele]


    if best_score<curr_score:
        best_spins=spins


    return  max(best_score,curr_score),best_spins,number_of_queries,curr_score,spins

@njit
def random_greedy(adj_matrix, weight_matrix, start_list, end_list,size_constraint):

    number_of_queries=0
    n=len(start_list)
    merginal_gain=np.zeros(n)
    spins=np.zeros(n)


    for i in range(n):
        number_of_queries+=1
        for j,weight in zip(adj_matrix[start_list[i]:end_list[i]],
                  weight_matrix[start_list[i]:end_list[i]]):

            merginal_gain[i]+=weight*(2*spins[i]-1)*(2*spins[j]-1)




    curr_score=0
    step=0

    for _ in range(size_constraint):
        arg_indices=np.argsort(-merginal_gain)

        indices = [index for index in arg_indices if spins[index] == 0 and merginal_gain[index]>0]
        number_of_queries+=(n-np.sum(spins))
        indices=indices[:size_constraint]
        len_indices=len(indices)



        if indices:
            rand_idx=np.random.randint(len_indices)
            rand_ele=indices[rand_idx]
            curr_score+=merginal_gain[rand_ele]

            merginal_gain[rand_ele]=-merginal_gain[rand_ele]
            for u,weight in zip(adj_matrix[start_list[rand_ele]:end_list[rand_ele]],
                            weight_matrix[start_list[rand_ele]:end_list[rand_ele]]):


                merginal_gain[u]+=weight*(2*spins[u]-1)*(2-4*spins[rand_ele])
            spins[rand_ele] = 1-spins[rand_ele]
        else:
            break



    return curr_score,spins,number_of_queries

@njit
def apx_local_search(adj_matrix, weight_matrix, start_list, end_list,size_constraint,error_rate,ground_set):

    n=len(start_list) # number of nodes
    merginal_gain=np.zeros(n)
    spins=np.zeros(n)

    number_of_queries=0

    # Calculate merginal gain for every element in ground set
    for i in range(n):
        for j,weight in zip(adj_matrix[start_list[i]:end_list[i]],
                         weight_matrix[start_list[i]:end_list[i]]):

            merginal_gain[i]+=weight*(2*spins[i]-1)*(2*spins[j]-1)
    number_of_queries+=np.sum(ground_set)
    # an approximation result

    max_gain=0
    A_0=-1
    for element in range(n):
        if ground_set[element]==1:
            if max_gain<merginal_gain[element]:
                A_0=element
                max_gain=merginal_gain[element]
    k=1
    curr_score=merginal_gain[A_0]

    merginal_gain[A_0]=-merginal_gain[A_0]

    for neighbour,weight in zip(adj_matrix[start_list[A_0]:end_list[A_0]],
                     weight_matrix[start_list[A_0]:end_list[A_0]]):

        merginal_gain[neighbour]+=weight*(2*spins[neighbour]-1)*(2-4*spins[A_0])
    spins[A_0]=1-spins[A_0]


    # EXCHANGE or DELETE

    continue_search=True


    while continue_search:

        best_spins=spins.copy()

        continue_search=False


        # EXCHANGE WITH DUMMY

        if k<size_constraint and not continue_search:
            for i in range(n):
                if spins[i] == 0 and ground_set[i]==1:
                    number_of_queries+=1
                    if merginal_gain[i]>=(error_rate/size_constraint**4)*curr_score:
                        continue_search=True
                        curr_score+=merginal_gain[i]
                        merginal_gain[i]=-merginal_gain[i]
                        for u,weight in zip(adj_matrix[start_list[i]:end_list[i]],
                                             weight_matrix[start_list[i]:end_list[i]]):

                            merginal_gain[u]+=weight*(2*spins[u]-1)*(2-4*spins[i])
                        spins[i] = 1-spins[i]
                        k+=1
                        break



        #EXCHANGE
        merginal_gain_copy=np.copy(merginal_gain)
        spins_copy=np.copy(spins)

        for e in range(n):

            if continue_search==True:
                break

            merginal_gain=np.copy(merginal_gain_copy)
            spins=np.copy(spins_copy)

            # In the solution set
            if spins[e]==1 and ground_set[e]==1:
                new_score=curr_score+merginal_gain[e] # (f(A-e))
                merginal_gain[e]=-merginal_gain[e]


                for u,weight in zip(adj_matrix[start_list[e]:end_list[e]],
                                     weight_matrix[start_list[e]:end_list[e]]):
                    merginal_gain[u]+=weight*(2*spins[u]-1)*(2-4*spins[e])
                spins[e] = 1-spins[e]


                for a in range(n):
                    if spins[a]==0 and ground_set[a]==1:
                        number_of_queries+=1 # (f(A-e+a))
                        if (new_score+merginal_gain[a])-curr_score>=(error_rate/size_constraint**4)*curr_score:
                            continue_search=True
                            #update
                            curr_score=new_score+merginal_gain[a]
                            merginal_gain[a]=-merginal_gain[a]
                            for u,weight in zip(adj_matrix[start_list[a]:end_list[a]],
                                            weight_matrix[start_list[a]:end_list[a]]):
                                merginal_gain[u]+=weight*(2*spins[u]-1)*(2-4*spins[a])
                            spins[a] = 1-spins[a]
                            break

        # DELETE
        if continue_search is False:
            merginal_gain=np.copy(merginal_gain_copy)
            spins=np.copy(spins_copy)

            for d in range(n):
                if spins[d]==1 and ground_set[i]==1:
                    number_of_queries+=1
                    if merginal_gain[d]>=(error_rate/size_constraint**4)*curr_score:
                        continue_search=True
                        curr_score+=merginal_gain[d]
                        merginal_gain[d]=-merginal_gain[d]
                        for u,weight in zip(adj_matrix[start_list[d]:end_list[d]],weight_matrix[start_list[d]:end_list[d]]):

                            merginal_gain[u]+=weight*(2*spins[u]-1)*(2-4*spins[d])
                        spins[d] = 1-spins[d]
                        k-=1
                        break




    return  curr_score,best_spins,number_of_queries

@njit
def lee_ls(adj_matrix, weight_matrix, start_list, end_list,size_constraint,error_rate):

    n=len(start_list)
    ground_set=np.ones(n,dtype=np.int32)


    best_score_1,spins_1,number_of_queries_1=apx_local_search(adj_matrix, weight_matrix, start_list,
                                                              end_list,size_constraint,error_rate,ground_set)

    for i in range(n):
        ground_set[i]=ground_set[i]-spins_1[i]
#     ground_set=ground_set-spins_1
    best_score_2,spins_2,number_of_queries_2=apx_local_search(adj_matrix, weight_matrix, start_list,
                                                              end_list,size_constraint,error_rate,ground_set)

    number_of_queries=number_of_queries_1+number_of_queries_2


    if best_score_1>best_score_2:
        return best_score_1,spins_1,number_of_queries
    else:
        return best_score_2,spins_2,number_of_queries

@njit
def get_adj_matrix(adj_matrix, weight_matrix, start_list, end_list ):

    n=len(start_list)

    G=np.zeros((n,n))

    for i in range(n):
        for j,weight in zip(adj_matrix[start_list[i]:end_list[i]],
                     weight_matrix[start_list[i]:end_list[i]]):
            G[i,j]=weight


    return G







def process_graph(graph_no,arg):
    
    model=args.model
    number_of_nodes = arg.n
    m=args.m 
    p=args.p
    k=args.k


    G = graph_gen(model=model, n=number_of_nodes, p=p, m=m, k=k)

    save_folder=f'data/Maximum Cut/{model}'
    os.makedirs(save_folder,exist_ok=True)
    
    filename=f'{model}{number_of_nodes}_graph{str(graph_no).zfill(3)}.npy'
    save_file_path=os.path.join(save_folder,filename)
    sparse_matrix = csr_matrix(G)
    save_npz(save_file_path, sparse_matrix)
    # np.save(file_path,G)
    adj_matrix, weight_matrix, start_list, end_list = flatten_graph(G)



    df=defaultdict(list)
    for mul in range(10,510,50):

        size_constraint=int(number_of_nodes*mul/1000)


        ls_sol, ls_spin,ls_queries= ls_greedy(adj_matrix, weight_matrix, start_list, end_list, size_constraint=size_constraint)
        fls_sol, fls_spin,fls_queries,guided_sol,guided_spin = fls_greedy(adj_matrix, weight_matrix,
                                                                         start_list, end_list,
                                                                         size_constraint=size_constraint,
                                                                         error_rate=0.01)
        lee_sol, lee_spin, lee_queries = lee_ls(adj_matrix, weight_matrix, start_list,
                                                                 end_list, size_constraint=size_constraint,
                                                               error_rate=0.1)

        rand_sol, rand_spin,rand_queries= random_greedy(adj_matrix, weight_matrix, start_list,
                                                                 end_list, size_constraint=size_constraint
                                                               )


        df['graph no'].append(graph_no)
        df['k'].append(size_constraint)
        df['greedy'].append(ls_sol)
        df['fls'].append(fls_sol)
        df['lee'].append(lee_sol)
        df['rand'].append(rand_sol)

        df['greedy_quries'].append(ls_queries)
        df['fls_quries'].append(fls_queries)
        df['lee_quries'].append(lee_queries)
        df['rand_quries'].append(rand_queries)

    df=pd.DataFrame(df)
    df.to_pickle(f"Maximum Cut/{model}/{graph_no}.pkl")






if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--model", type=str,default="BA", help="Distribution of dataset")
    parser.add_argument("--n", type=int, default=10000, help="the number of nodes")
    parser.add_argument("--m", type=int, default=4, help="m")
    parser.add_argument("--p", type=int, default=0.001, help="p")
    parser.add_argument("--k", type=int, default=10, help="k")
    args = parser.parse_args()
    
    os.makedirs(f'Maximum Cut/{args.model}', exist_ok = True)



    start_time = time.perf_counter()


    number_of_graphs = 20
    print(f"Number of graphs:{number_of_graphs}")
    processes = []

    for graph_no in range(number_of_graphs):
        process = Process(target=process_graph, args=(graph_no,args))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

