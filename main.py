#!/usr/bin/python
# -*- coding: UTF-8 -*-
import networkx as nx
import random as ran
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from pathos.pools import ProcessPool, ThreadPool
from tqdm import tqdm
import os
import sys


TURN_MAX = 10
SAMPING_MAX = 1000
THEORY_SAMPING_MAX = TURN_MAX * SAMPING_MAX
ALPHA = -0.05 # just for SDM
ALL_METHODS = ['CN', 'PA', 'SDM', 'Salton', 'Jaccard', 'Sorensen', 'HPI', 'HDI', 'LHNI']
FIRST_ORDER_METHOD = ['CN']
SECOND_ORDER_METHOD = ['PA', 'SDM']
THIRD_ORDER_METHOD = ['Salton', 'Jaccard', 'Sorensen', 'HPI', 'HDI', 'LHNI']
N_MAX = 500
ALL_DEGREE_TYPE = ['power-law', 'exponential', 'normal', 'lognormal']
METHOD = FIRST_ORDER_METHOD + SECOND_ORDER_METHOD + THIRD_ORDER_METHOD
undirected_document = "./dataset/newUndirected"
directed_document = './dataset/newDirected'
output_file = 'delta.txt'
rewiring_file = 'rewiring.txt'
all_rewires = ['non', 'zero', 'first', 'first_ass', 'first_dis', 'first_ass_constCC', 'first_dis_constCC', 'second', 'third', 'third_increaseCC', 'third_decreaseCC']
artificial_output_file = './undirected_artificial_auc.txt'
rewiring_num = 10000


def parallel(func, *args, show=False, thread=False, **kwargs):
    p_func = partial(func, **kwargs)
    pool = ThreadPool() if thread else ProcessPool()
    try:
        if show:
            start = time.time()
            # imap
            with tqdm(total=len(args[0]), desc="计算进度") as t:  # 进度条设置
                r = []
                for i in pool.imap(p_func, *args):
                    r.append(i)
                    t.set_postfix({'并行函数': func.__name__, "计算花销": "%ds" % (time.time() - start)})
                    t.update()
        else:
            # map
            r = pool.map(p_func, *args)
        return r
    except Exception as e:
        print(e)
    finally:
        pool.close()  # close the pool to any new jobs
        pool.join()  # cleanup the closed worker processes
        pool.clear()  # Remove server with matching state


def link_prediction_system(g1, division, algorithm, RAND_PROB, measurement='auc'):
  if nx.is_directed(g1):
    network = nx.DiGraph()
  else:
    network = nx.Graph()
  network.add_edges_from(list(g1.edges()))
  # parameter statistics
  N, M = network.number_of_nodes(), network.number_of_edges()
  all_nodes = list(network.nodes())
  score_index  = 0

  # independent replicate experiments
  for turn in range(TURN_MAX):

    # network data division processing
    if division == 'rand':
      training_links, probe_links = random_division(network, RAND_PROB)
    elif division == 'fixed':
      training_links, probe_links = fixed_strategy(network, M * RAND_PROB)
    
    in_neighbors = dict(zip(all_nodes, [[] for node in all_nodes]))
    out_neighbors = in_neighbors
    for edge in training_links:
      in_neighbors[edge[1]].append(edge[0])
      out_neighbors[edge[0]].append(edge[1])
      if not nx.is_directed(network):
        in_neighbors[edge[0]].append(edge[1])
        out_neighbors[edge[1]].append(edge[0])

    # list all scoring methods
    if algorithm == 'CN':
      score = lambda x, y: len(list(set(out_neighbors[x]) & set(out_neighbors[y])))
    elif algorithm == 'PA':
      score = lambda x, y: len(out_neighbors[x]) * len(out_neighbors[y])
    elif algorithm == 'SDM':
      score = lambda x, y: 1.0 / (math.exp(ALPHA * len(out_neighbors[x])) + math.exp(ALPHA * len(out_neighbors[y])))
    elif algorithm == 'Salton':
      score = lambda x, y: len(list(set(out_neighbors[x]) & set(out_neighbors[y]))) / math.sqrt(len(out_neighbors[x]) * len(out_neighbors[y]))
    elif algorithm == 'Jaccard':
      score = lambda x, y: len(list(set(out_neighbors[x]) & set(out_neighbors[y]))) / (len(out_neighbors[x]) + len(out_neighbors[y]) - len(list(set(out_neighbors[x]) & set(out_neighbors[y]))))
    elif algorithm == 'Sorensen':
      score = lambda x, y: len(list(set(out_neighbors[x]) & set(out_neighbors[y]))) / math.sqrt(len(out_neighbors[x]) + len(out_neighbors[y]))
    elif algorithm == 'HPI':
      score = lambda x, y: len(list(set(out_neighbors[x]) & set(out_neighbors[y]))) / min(len(out_neighbors[x]), len(out_neighbors[y]))
    elif algorithm == 'HDI':
      score = lambda x, y: len(list(set(out_neighbors[x]) & set(out_neighbors[y]))) / max(len(out_neighbors[x]), len(out_neighbors[y]))
    elif algorithm == 'LHNI':
      score = lambda x, y: len(list(set(out_neighbors[x]) & set(out_neighbors[y]))) / (len(out_neighbors[x]) * len(out_neighbors[y]))

    # sampling, scoring and measuring the performance
    sampling_time = 0
    if measurement == 'auc':
      while sampling_time < SAMPING_MAX:
        node1, node2 = ran.choice(all_nodes), ran.choice(all_nodes)
        if node1 == node2 or network.has_edge(node1, node2):
          continue
        sampling_time += 1
        node3, node4 = ran.choice(probe_links)

        try:
          nonexist_score = score(node1, node2)
        except:
          nonexist_score = 0
        
        try:
          probe_score = score(node3, node4)
        except:
          probe_score = 0

        if nonexist_score < probe_score:
          score_index += 1.0
        elif nonexist_score == probe_score:
          score_index += 0.5

  return score_index / TURN_MAX / SAMPING_MAX


def random_division(network, p):
  train, probe = [], []
  for edge in network.edges():
    if ran.random() < p:
      probe.append(edge)
    else:
      train.append(edge)
  return train, probe


def fixed_strategy(network, m):
  probe = []
  train = list(network.edges())
  for i in range(m):
    edge = ran.choice(train)
    probe.append(edge)
    train.remove(edge)
  return train, probe


def auc_theoretical_analysis(network, algorithm, RAND_PROB):
  N, M = network.number_of_nodes(), network.number_of_edges()
  all_nodes = list(network.nodes())
  all_edges = list(network.edges())

  in_neighbors = dict(zip(all_nodes, [[] for node in all_nodes]))
  out_neighbors = in_neighbors
  for edge in network.edges():
    in_neighbors[edge[1]].append(edge[0])
    out_neighbors[edge[0]].append(edge[1])
    if not nx.is_directed(network):
      in_neighbors[edge[0]].append(edge[1])
      out_neighbors[edge[1]].append(edge[0])

  sampling_index = 0
  score_index = 0.0
  while sampling_index < THEORY_SAMPING_MAX:
    node1, node2 = ran.choice(all_nodes), ran.choice(all_nodes)
    if node1 == node2 or network.has_edge(node1, node2):
      continue
    sampling_index += 1
    node3, node4 = ran.choice(all_edges)

    if algorithm in FIRST_ORDER_METHOD:
      original_nonexist_topology = len(list(set(out_neighbors[node1]) & set(out_neighbors[node2])))
      original_exist_topology = len(list(set(out_neighbors[node3]) & set(out_neighbors[node4])))
      
      nonexist_topology = original_nonexist_topology
      probe_topology = original_exist_topology

      if original_nonexist_topology > 0:
        for random_division_turn in range(original_nonexist_topology):
          if ran.random() > (1 - RAND_PROB) ** 2:
            nonexist_topology -= 1
      if original_exist_topology > 0:
        for random_division_turn in range(original_exist_topology):
          if ran.random() > (1 - RAND_PROB) ** 2:
            probe_topology -= 1
      
      nonexist_score, probe_score = nonexist_topology, probe_topology

      if nonexist_score < probe_score:
        score_index += 1
      elif nonexist_score == probe_score:
        score_index += 0.5
    
    if algorithm in SECOND_ORDER_METHOD:
      original_nonexist_topology = [len(out_neighbors[node1]), len(out_neighbors[node2])]
      original_exist_topology = [len(out_neighbors[node3]), len(out_neighbors[node4])]
      
      nonexist_topology = [x for x in original_nonexist_topology]
      probe_topology = [x - 1 for x in original_exist_topology]

      for i in range(2):
        if original_nonexist_topology[i] > 0:
          for random_division_turn in range(original_nonexist_topology[i]):
            if ran.random() < RAND_PROB:
              nonexist_topology[i] -= 1
        if original_exist_topology[i] > 1:
          for random_division_turn in range(original_exist_topology[i] - 1):
            if ran.random() < RAND_PROB:
              probe_topology[i] -= 1

      if algorithm == 'PA':
        nonexist_score = nonexist_topology[0] * nonexist_topology[1]
        probe_score = probe_topology[0] * probe_topology[1]
      elif algorithm == 'SDM':
        nonexist_score = 1.0 / (math.exp(ALPHA * nonexist_topology[0]) + math.exp(ALPHA * nonexist_topology[1]))
        probe_score = 1.0 / (math.exp(ALPHA * probe_topology[0]) + math.exp(ALPHA * probe_topology[1]))

      if nonexist_score < probe_score:
        score_index += 1
      elif nonexist_score == probe_score:
        score_index += 0.5
  
    if algorithm in THIRD_ORDER_METHOD:
      original_exist_topology = [
        len(out_neighbors[node3]) - len(list(set(out_neighbors[node3]) & set(out_neighbors[node4]))), 
        len(out_neighbors[node4]) - len(list(set(out_neighbors[node3]) & set(out_neighbors[node4]))),
        len(list(set(out_neighbors[node3]) & set(out_neighbors[node4])))
        ]
      original_nonexist_topology = [
        len(out_neighbors[node1]) - len(list(set(out_neighbors[node1]) & set(out_neighbors[node2]))), 
        len(out_neighbors[node2]) - len(list(set(out_neighbors[node1]) & set(out_neighbors[node2]))),
        len(list(set(out_neighbors[node1]) & set(out_neighbors[node2])))
        ]
      nonexist_topology = [x for x in original_nonexist_topology]
      probe_topology = [x for x in original_exist_topology]
      for i in range(2):
        probe_topology[i] -= 1

      for i in range(3):
        if original_nonexist_topology[i] > 0:
          if i < 2:
            for random_division_turn in range(original_nonexist_topology[i]):
              if ran.random() < RAND_PROB:
                nonexist_topology[i] -= 1
          else:
            for random_division_turn in range(original_nonexist_topology[i]):
              if ran.random() > (1 - RAND_PROB) ** 2:
                nonexist_topology[i] -= 1

        if i < 2 and original_exist_topology[i] > 1:
          for random_division_turn in range(original_exist_topology[i] - 1):
            if ran.random() < RAND_PROB:
              probe_topology[i] -= 1
        elif i == 2 and original_exist_topology[i] > 0:
          for random_division_turn in range(original_exist_topology[i]):
            if ran.random() > (1 - RAND_PROB) ** 2:
              probe_topology[i] -= 1
      
      if algorithm == 'Salton':
        try:
          nonexist_score = nonexist_topology[2] / math.sqrt((nonexist_topology[0] + nonexist_topology[2]) * (nonexist_topology[1] + nonexist_topology[2]))
        except:
          nonexist_score = 0
        
        try:
          probe_score = probe_topology[2] / math.sqrt((probe_topology[0] + probe_topology[2]) * (probe_topology[1] + probe_topology[2]))
        except:
          probe_score = 0

      elif algorithm == 'Jaccard':
        try:
          nonexist_score = nonexist_topology[2] / sum(nonexist_topology)
        except:
          nonexist_score = 0

        try:
          probe_score = probe_topology[2] / sum(probe_topology)
        except:
          probe_score = 0
      
      elif algorithm == 'Sorensen':
        try:
          nonexist_score = nonexist_topology[2] / (sum(nonexist_topology) + nonexist_topology[2])
        except:
          nonexist_score = 0

        try:
          probe_score = probe_topology[2] / (sum(probe_topology) + probe_topology[2])
        except:
          probe_score = 0
      
      elif algorithm == 'HPI':
        try:
          nonexist_score = nonexist_topology[2] / (min(nonexist_topology[:2]) + nonexist_topology[2])
        except:
          nonexist_score = 0

        try:
          probe_score = probe_topology[2] / (min(probe_topology[:2]) + probe_topology[2])
        except:
          probe_score = 0
      
      elif algorithm == 'HDI':
        try:
          nonexist_score = nonexist_topology[2] / (max(nonexist_topology[:2]) + nonexist_topology[2])
        except:
          nonexist_score = 0

        try:
          probe_score = probe_topology[2] / (max(probe_topology[:2]) + probe_topology[2])
        except:
          probe_score = 0
      
      elif algorithm == 'LHNI':
        try:
          nonexist_score = nonexist_topology[2] / ((nonexist_topology[0] + nonexist_topology[2]) * (nonexist_topology[1] + nonexist_topology[2]))
        except:
          nonexist_score = 0

        try:
          probe_score = probe_topology[2] / ((probe_topology[0] + probe_topology[2]) * (probe_topology[1] + probe_topology[2]))
        except:
          probe_score = 0
      
      if nonexist_score < probe_score:
        score_index += 1
      elif nonexist_score == probe_score:
        score_index += 0.5
  return score_index / THEORY_SAMPING_MAX


def read_network(file_path, net_type):
  if net_type == 'undirected':
    g = nx.Graph()
  elif net_type == 'directed':
    g = nx.DiGraph()

  with open(file_path, encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
      if '\t' in line:
        line1 = line.strip('\n').split('\t')
      else:
        line1 = line.strip('\n').split(' ')
      if g.has_edge(line1[1], line1[0]) or g.has_edge(line1[0], line1[1]):
        continue
      g.add_edge(line1[0], line1[1])
  if net_type == 'undirected':
    return g.subgraph(max(nx.connected_components(g), key=len))
  elif net_type == 'directed':
    return g.subgraph(max(nx.weakly_connected_components(g), key=len))


def LIC(g, delta_sample):
  nodes = list(g.nodes())
  edges = list(g.edges())
  sampling_index = 0
  dependent_index = 0.0
  while sampling_index < delta_sample:
    node1, node2 = ran.choice(nodes), ran.choice(nodes)
    if node1 == node2 or g.has_edge(node1, node2):
      continue
    sampling_index += 1
    edge = ran.choice(edges)
    if node1 in edge or node2 in edge:
      dependent_index += 1.0
    elif g.has_edge(node1, edge[0]) or g.has_edge(node2, edge[0]) or g.has_edge(node1, edge[1]) or g.has_edge(node2, edge[1]) or g.has_edge(edge[0], node1) or g.has_edge(edge[0], node2) or g.has_edge(edge[1], node1) or g.has_edge(edge[1], node2):
      dependent_index += 1.0
  
  return dependent_index / delta_sample


def output(file_name, net_type, all_delta_sample=100000):
  G = read_network(file_name, net_type)
  # rewire(G, minimum_rewire_turn=20, rewire_type=specific_type)
  # print('generating...')
  N, M = G.number_of_nodes(), G.number_of_edges()
  CC = round(nx.average_clustering(G),4)
  try:
    r = round(nx.degree_assortativity_coefficient(G),4)
  except:
    r = 2
  l = LIC(G, all_delta_sample)

  classical_simulations = []
  novel_theorys = []
  # print(file_name, N, M, CC, r)
  for method1 in METHOD:
    for s in range(1, 10):
      rand_number = s * 0.1
      classical_simulations.append(str(round(link_prediction_system(G, 'rand', method1, rand_number),4)))
      novel_theorys.append(str(round(auc_theoretical_analysis(G, method1, rand_number),4)))
  
  if net_type == 'undirected':
    file_name = file_name[len(undirected_document)+1:]
  else:
    file_name = file_name[len(directed_document)+1:]
  print('topology:', file_name, N, M, CC, r, l, sep='\t')
  print('old:\t', '\t'.join(classical_simulations))
  print('new:\t', '\t'.join(novel_theorys))
  with open(net_type+'_'+output_file, 'a') as f:
    f.write('\t'.join([file_name, str(N), str(M), str(CC), str(r), str(l)] + classical_simulations + novel_theorys))
    f.write('\n')


def rand_ER(g1, connected):
  if nx.is_directed(g1):
    g = nx.DiGraph(list(g1.edges()))
  else:
    g = nx.Graph(list(g1.edges()))
  rewiring_index = 0
  error_index = 0
  if not nx.is_directed(g):
    while rewiring_index < rewiring_num and error_index < 10000:
      # print(rewiring_index, error_index)
      node1 = ran.choice(list(g.nodes()))
      node2 = ran.choice(list(g.nodes()))
      if node1 == node2 or g.has_edge(node1, node2) or g.has_edge(node2, node1):
        error_index += 1
        continue
      edge = ran.choice(list(g.edges()))
      g.remove_edge(edge[0], edge[1])
      g.add_edge(node1, node2)
      # if connected:
      #   if not nx.is_connected(g):
      #     g.add_edge(edge[0], edge[1])
      #     g.remove_edge(node1, node2)
      #     continue
      rewiring_index += 1
      error_index = 0
  else:
    while rewiring_index < rewiring_num and error_index < 10000:
      # print(rewiring_index, error_index)
      node1 = ran.choice(list(g.nodes()))
      node2 = ran.choice(list(g.nodes()))
      if node1 == node2 or g.has_edge(node1, node2):
        error_index += 1
        continue
      edge = ran.choice(list(g.edges()))
      g.remove_edge(edge[0], edge[1])
      g.add_edge(node1, node2)
      # if connected:
      #   if not nx.is_weakly_connected(g):
      #     g.add_edge(edge[0], edge[1])
      #     g.remove_edge(node1, node2)
      #     continue
      rewiring_index += 1
      error_index = 0
  if error_index >= 10000:
    return False
  else:
    return g


def model_rand_ER(g1):
  n, m = g1.number_of_nodes(), g1.number_of_edges()
  if nx.is_directed(g1):
    p = 1.0*m/n/(n-1)
    return nx.erdos_renyi_graph(n, p, directed=True)
  else:
    p = 2.0*m/n/(n-1)
    return nx.erdos_renyi_graph(n, p, directed=False)


def rand_deg(g1, connected):
  if nx.is_directed(g1):
    g = nx.DiGraph(list(g1.edges()))
  else:
    g = nx.Graph(list(g1.edges()))
  rewiring_index = 0
  error_index = 0
  if not nx.is_directed(g):
    while rewiring_index < rewiring_num and error_index < 10000:
      # print(rewiring_index, error_index)
      node1, node2 = ran.choice(list(g.edges()))
      node3, node4 = ran.choice(list(g.edges()))
      if ran.random() < 0.5:
        node1, node2 = node2, node1
      if node1 == node3 or g.has_edge(node3, node1) or g.has_edge(node1, node3):
        error_index += 1
        continue
      if node2 == node4 or g.has_edge(node2, node4) or g.has_edge(node4, node2):
        error_index += 1
        continue
      g.remove_edge(node1, node2)
      g.remove_edge(node3, node4)
      g.add_edge(node1, node3)
      g.add_edge(node2, node4)
      # if connected:
      #   if not nx.is_connected(g):
      #     g.add_edge(node1, node2)
      #     g.add_edge(node3, node4)
      #     g.remove_edge(node1, node3)
      #     g.remove_edge(node2, node4)
      #     error_index += 1
      #     continue
      rewiring_index += 1
      error_index = 0
  else:
    while rewiring_index < rewiring_num and error_index < 10000:
      # print(rewiring_index, error_index)
      node1, node2 = ran.choice(list(g.edges()))
      node3, node4 = ran.choice(list(g.edges()))
      if node1 == node4 or g.has_edge(node1, node4):
        error_index += 1
        continue
      if node2 == node3 or g.has_edge(node3, node2):
        error_index += 1
        continue
      g.remove_edge(node1, node2)
      g.remove_edge(node3, node4)
      g.add_edge(node1, node4)
      g.add_edge(node3, node2)
      # if connected:
      #   if not nx.is_weakly_connected(g):
      #     g.add_edge(node1, node2)
      #     g.add_edge(node3, node4)
      #     g.remove_edge(node1, node4)
      #     g.remove_edge(node3, node2)
      #     error_index += 1
      #     continue
      rewiring_index += 1
      error_index = 0
  
  if error_index >= 10000:
    return False
  else:
    return g
  

def model_rand_deg(g1):
  if nx.is_directed(g1):
    out_degree_seq = dict(g1.out_degree()).values()
    in_degree_seq = dict(g1.in_degree()).values()
    g = nx.directed_configuration_model(in_degree_seq, out_degree_seq)
    g = nx.DiGraph(g)
    g.remove_edges_from(nx.selfloop_edges(g))
  else:
    degree_seq = dict(g1.degree()).values()
    g = nx.configuration_model(degree_seq)
    g = nx.Graph(g)
    g.remove_edges_from(nx.selfloop_edges(g))
  return g


def rand_deg_deg(g1, connected):
  if nx.is_directed(g1):
    g = nx.DiGraph(list(g1.edges()))
  else:
    g = nx.Graph(list(g1.edges()))
  out_deg2nodes = {}
  rewiring_index = 0
  error_index = 0
  if not nx.is_directed(g):
    for node in g.nodes():
      d = g.degree(node)
      if d not in out_deg2nodes:
        out_deg2nodes[d] = []
      out_deg2nodes[d].append(node)
    while rewiring_index < rewiring_num and error_index < 10000:
      # print(rewiring_index, error_index)
      node1, node2 = ran.choice(list(g.edges()))
      if ran.random() < 0.5:
        node1, node2 = node2, node1
      node3 = ran.choice(out_deg2nodes[g.degree(node1)])
      if node1 == node3:
        error_index += 1
        continue
      node4 = ran.choice(list(g[node3]))
      if node2 == node4:
        error_index += 1
        continue
      if g.has_edge(node1, node4) or g.has_edge(node4, node1) or g.has_edge(node3, node2) or g.has_edge(node2, node3):
        error_index += 1
        continue
      g.remove_edge(node1, node2)
      g.remove_edge(node3, node4)
      g.add_edge(node1, node4)
      g.add_edge(node3, node2)
      # if connected:
      #   if not nx.is_connected(g):
      #     g.add_edge(node1, node2)
      #     g.add_edge(node3, node4)
      #     g.remove_edge(node1, node4)
      #     g.remove_edge(node3, node2)
      #     error_index += 1
      #     continue
      rewiring_index += 1
      error_index = 0
  else:
    for node in g.nodes():
      d = g.out_degree(node)
      if d not in out_deg2nodes:
        out_deg2nodes[d] = []
      out_deg2nodes[d].append(node)
    while rewiring_index < rewiring_num and error_index < 10000:
      # print(rewiring_index, error_index)
      node1, node2 = ran.choice(list(g.edges()))
      node3 = ran.choice(out_deg2nodes[g.out_degree(node1)])
      if node1 == node3:
        error_index += 1
        continue
      node4 = ran.choice(list(g[node3]))
      if node2 == node4:
        error_index += 1
        continue
      if g.has_edge(node1, node4) or g.has_edge(node3, node2):
        error_index += 1
        continue
      g.remove_edge(node1, node2)
      g.remove_edge(node3, node4)
      g.add_edge(node1, node4)
      g.add_edge(node3, node2)
      # if connected:
      #   if not nx.is_connected(g):
      #     g.add_edge(node1, node2)
      #     g.add_edge(node3, node4)
      #     g.remove_edge(node1, node4)
      #     g.remove_edge(node3, node2)
      #     error_index += 1
      #     continue
      rewiring_index += 1
      error_index = 0
  if error_index >= 10000:
    return False
  else:
    return g


def theory_and_classical_aucs(test):
  with open('undirected'+'_'+output_file, 'w') as f:
    None
  with open('directed'+'_'+output_file, 'w') as f:
    None
  
  if test:
    # Compute two examplem datasets in parallel
    output(undirected_document + '/10.txt', 'undirected')
    output(directed_document + '/14.txt', 'directed')
  else:
    # Compute multiple datasets in parallel
    fs = []
    nts = []
    for file in os.listdir(undirected_document):
      fs.append(undirected_document + '/' + file)
      nts.append('undirected')
    for file in os.listdir(directed_document):
      fs.append(directed_document + '/' + file)
      nts.append('directed')
    parallel(output, fs, nts)


def rewiring_output(file_name, net_type, connected):
  rewiring = []
  print(file_name, 'begining...')
  G = read_network(file_name, net_type)
  print(G.number_of_nodes(), G.number_of_edges())
  # G1 = rand_ER(G, connected)
  G1 = model_rand_ER(G)
  if G1:
    for method1 in METHOD:
      for s in range(1, 10):
        rand_number = s * 0.1
        r1 = round(link_prediction_system(G1, 'rand', method1, rand_number),4)
        rewiring.append(str(r1))
    print(file_name + " rand-ER success!")
  else:
    rewiring += ['-1' for ix in range(81)]
    print(file_name + " rand-ER fail!")

  G2 = model_rand_deg(G)
  if G2:
    for method1 in METHOD:
      for s in range(1, 10):
        rand_number = s * 0.1
        r2 = round(link_prediction_system(G2, 'rand', method1, rand_number),4)
        rewiring.append(str(r2))
    print(file_name + " rand-deg success!")
  else:
    rewiring += ['-1' for ix in range(81)]
    print(file_name + " rand-deg fail!")

  G3 = rand_deg_deg(G, connected)
  if G3:
    for method1 in METHOD:
      for s in range(1, 10):
        rand_number = s * 0.1
        r3 = round(link_prediction_system(G3, 'rand', method1, rand_number),4)
        rewiring.append(str(r3))
    print(file_name + " rand-deg-deg success!")
  else:
    rewiring += ['-1' for ix in range(81)]
    print(file_name + " rand-deg-deg fail!")
      
  if net_type == 'undirected':
    file_name = file_name[len(undirected_document)+1:]
  else:
    file_name = file_name[len(directed_document)+1:]
  with open(net_type+'_'+rewiring_file, 'a') as f:
    f.write('\t'.join([file_name] + rewiring))
    f.write('\n')


def rewiring_classical_auc(test, connected):
  with open('undirected'+'_'+rewiring_file, 'w') as f:
    None
  with open('directed'+'_'+rewiring_file, 'w') as f:
    None
  if test:
    rewiring_output(undirected_document + '/114.txt', 'undirected', connected)
    rewiring_output(directed_document + '/121.txt', 'directed', connected)
  else:
    fs = []
    nts = []
    cts = []
    for file in os.listdir(undirected_document):
      fs.append(undirected_document + '/' + file)
      nts.append('undirected')
      cts.append(connected)
    for file in os.listdir(directed_document):
      fs.append(directed_document + '/' + file)
      nts.append('directed')
      cts.append(connected)
    parallel(rewiring_output, fs, nts, cts)
  

if __name__ == '__main__':
  module_num = int(sys.argv[1])
  def f(x):
    if x == '0':
      return False
    else:
      return True

  if module_num == 0:
    theory_and_classical_aucs(f(sys.argv[2]))
  elif module_num == 1:
    rewiring_classical_auc(f(sys.argv[2]), f(sys.argv[3]))


