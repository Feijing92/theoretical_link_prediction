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
import gc


TURN_MAX = 20
SAMPING_MAX = 2000
THEORY_SAMPING_MAX = TURN_MAX * SAMPING_MAX
ALPHA = -0.05 # just for SDM
ALL_METHODS = ['CN', 'PA', 'SDM', 'Salton', 'Jaccard', 'Sorensen', 'HPI', 'HDI', 'LHNI']
FIRST_ORDER_METHOD = ['CN']
SECOND_ORDER_METHOD = ['PA', 'SDM']
THIRD_ORDER_METHOD = ['Salton', 'Jaccard', 'Sorensen', 'HPI', 'HDI', 'LHNI']
ERROR_MAX = 100000
ALL_DEGREE_TYPE = ['power-law', 'exponential', 'normal', 'lognormal']
METHOD = FIRST_ORDER_METHOD + SECOND_ORDER_METHOD + THIRD_ORDER_METHOD
undirected_document = './dataset/newUndirected'
directed_document = './dataset/newDirected'
undirected_artificial = './artificial_networks/undirected'
directed_artificial = './artificial_networks/directed'
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
  turn = 0
  # independent replicate experiments
  while turn < TURN_MAX:

    # network data division processing
    if division == 'rand':
      training_links, probe_links = random_division(network, RAND_PROB)
    elif division == 'fixed':
      training_links, probe_links = fixed_strategy(network, M * RAND_PROB)

    if len(probe_links) == 0:
      # score_index += 0.5 * SAMPING_MAX
      continue

    turn += 1
    
    out_neighbors = dict(zip(all_nodes, [[] for node in all_nodes]))
    for edge in training_links:
      out_neighbors[edge[0]].append(edge[1])
      if not nx.is_directed(network):
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
        if node1 == node2 or network.has_edge(node1, node2) or network.has_edge(node2, node1):
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

  # if algorithm == 'CN':
  #   print(probe_links, score_index / TURN_MAX / SAMPING_MAX)
  return score_index / TURN_MAX / SAMPING_MAX


def random_division(network, p):
  train, probe = [], []
  for edge in network.edges():
    if ran.random() < p:
      probe.append(edge)
    else:
      train.append(edge)
  # print(probe)
  return train, probe


def fixed_strategy(network, m):
  probe = []
  train = list(network.edges())
  for i in range(m):
    edge = ran.choice(train)
    probe.append(edge)
    train.remove(edge)
  return train, probe


def undirected_auc_theoretical_analysis(network, algorithm, RAND_PROB):
  N, M = network.number_of_nodes(), network.number_of_edges()
  all_nodes = list(network.nodes())
  all_edges = list(network.edges())

  out_neighbors = dict(zip(all_nodes, [[] for node in all_nodes]))
  for edge in network.edges():
    out_neighbors[edge[0]].append(edge[1])
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
      
      nonexist_topology = 0
      probe_topology = 0

      if original_nonexist_topology > 0:
        for random_division_turn in range(original_nonexist_topology):
          if ran.random() < (1 - RAND_PROB) ** 2:
            nonexist_topology += 1
      if original_exist_topology > 0:
        for random_division_turn in range(original_exist_topology):
          if ran.random() < (1 - RAND_PROB) ** 2:
            probe_topology += 1
      
      nonexist_score, probe_score = nonexist_topology, probe_topology

      if nonexist_score < probe_score:
        score_index += 1
      elif nonexist_score == probe_score:
        score_index += 0.5
    
    if algorithm in SECOND_ORDER_METHOD:
      original_nonexist_topology = [len(out_neighbors[node1]), len(out_neighbors[node2])]
      original_exist_topology = [len(out_neighbors[node3]), len(out_neighbors[node4])]
      
      nonexist_topology = [0, 0]
      probe_topology = [0, 0]

      for i, x in enumerate(original_nonexist_topology):
        if x > 0:
          for random_division_turn in range(x):
            if ran.random() > RAND_PROB:
              nonexist_topology[i] += 1
      
      for i, x in enumerate(original_exist_topology):
        if x > 1:
            for random_division_turn in range(x - 1):
              if ran.random() > RAND_PROB:
                probe_topology[i] += 1

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
      cn1 = len(list(set(out_neighbors[node3]) & set(out_neighbors[node4])))
      cn2 = len(list(set(out_neighbors[node1]) & set(out_neighbors[node2])))
      original_exist_topology = [len(out_neighbors[node3]) - cn1, len(out_neighbors[node4]) - cn1, cn1]
      original_nonexist_topology = [len(out_neighbors[node1]) - cn2, len(out_neighbors[node2]) - cn2, cn2]
      nonexist_topology = [0, 0, 0]
      probe_topology = [0, 0, 0]

      for i in range(3):
        if original_nonexist_topology[i] > 0:
          if i < 2:
            for random_division_turn in range(original_nonexist_topology[i]):
              if ran.random() > RAND_PROB:
                nonexist_topology[i] += 1
          else:
            for random_division_turn in range(original_nonexist_topology[i]):
              aa = [0, 0]
              if ran.random() < 1 - RAND_PROB:
                aa[0] = 1
              if ran.random() < 1 - RAND_PROB:
                aa[1] = 1
              
              if aa[0] + aa[1] == 2:
                nonexist_topology[2] += 1
              elif aa[0] == 1:
                nonexist_topology[0] += 1
              elif aa[1] == 1:
                nonexist_topology[1] += 1

        if i < 2 and original_exist_topology[i] > 1:
          for random_division_turn in range(original_exist_topology[i] - 1):
            if ran.random() > RAND_PROB:
              probe_topology[i] += 1
        elif i == 2 and original_exist_topology[i] > 0:
          for random_division_turn in range(original_exist_topology[i]):
            aa = [0, 0]
            if ran.random() < 1 - RAND_PROB:
              aa[0] = 1
            if ran.random() < 1 - RAND_PROB:
              aa[1] = 1
            
            if aa[0] + aa[1] == 2:
              probe_topology[2] += 1
            elif aa[0] == 1:
              probe_topology[0] += 1
            elif aa[1] == 1:
              probe_topology[1] += 1
      
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


def undirected_auc_theoretical_analysis_combined_with_motif(network, algorithm, RAND_PROB):
  N, M = network.number_of_nodes(), network.number_of_edges()
  all_nodes = list(network.nodes())
  all_edges = list(network.edges())

  out_neighbors = dict(zip(all_nodes, [[] for node in all_nodes]))
  for edge in network.edges():
    out_neighbors[edge[0]].append(edge[1])
    out_neighbors[edge[1]].append(edge[0])

  sampling_index = 0
  score_index = 0.0

  while sampling_index < THEORY_SAMPING_MAX:
    node1, node2 = ran.choice(all_nodes), ran.choice(all_nodes)
    if node1 == node2 or network.has_edge(node1, node2):
      continue
    sampling_index += 1
    node3, node4 = ran.choice(all_edges)
    motif = 0

    # dependency testing...
    
    if node1 == node3:
      motif = 1
      overlap_node = node1
      other_node1 = node2
      other_node2 = node4
    elif node1 == node4:
      motif = 1
      overlap_node = node1
      other_node1 = node2
      other_node2 = node3
    elif node2 == node3:
      motif = 1
      overlap_node = node2
      other_node1 = node1
      other_node2 = node4
    elif node2 == node4:
      motif = 1
      overlap_node = node2
      other_node1 = node1
      other_node2 = node3
    else:
      dependency = [0, 0, 0, 0]

      if network.has_edge(node1, node3):
        dependency[0] = 1
      if network.has_edge(node1, node4):
        dependency[1] = 1
      if network.has_edge(node2, node3):
        dependency[2] = 1
      if network.has_edge(node2, node4):
        dependency[3] = 1

      overlap = sum(dependency)
      if overlap > 0:
        motif = 2
    
    # conduct local samping algorithms with dependency cases
    
    if algorithm in FIRST_ORDER_METHOD:
      original_nonexist_common_neighbors = list(set(out_neighbors[node1]) & set(out_neighbors[node2]))
      original_nonexist_topology = len(original_nonexist_common_neighbors)

      original_exist_common_neighbors = list(set(out_neighbors[node3]) & set(out_neighbors[node4]))
      original_exist_topology = len(original_exist_common_neighbors)

      original_overlap = 0

      nonexist_topology = 0
      probe_topology = 0

      if motif == 1:
        original_overlap_common_neighbors = list(set(original_nonexist_common_neighbors) & set(original_exist_common_neighbors))
        original_overlap = len(original_overlap_common_neighbors)

        if node3 in original_nonexist_common_neighbors or node4 in original_nonexist_common_neighbors:
          original_nonexist_topology -= 1

        for random_division_turn in range(original_overlap):
          if ran.random() < 1 - RAND_PROB:
            if ran.random() < 1 - RAND_PROB:
              nonexist_topology += 1
            if ran.random() < 1 - RAND_PROB:
              probe_topology += 1
        
        original_nonexist_topology -= original_overlap
        original_exist_topology -= original_overlap
              
      elif motif == 2:

        if overlap == 3:
          if ran.random() < 1 - RAND_PROB:
            if ran.random() < 1 - RAND_PROB:
              nonexist_topology += 1
            if ran.random() < 1 - RAND_PROB:
              probe_topology += 1
          
          original_nonexist_topology -= 1
          original_exist_topology -= 1
        
        elif overlap == 4:
          for id in range(4):
            if ran.random() < RAND_PROB:
              dependency[id] = 0
          
          if dependency[0] * dependency[1] == 1:
            probe_topology += 1
          if dependency[0] * dependency[2] == 1:
            nonexist_topology += 1
          if dependency[2] * dependency[3] == 1:
            probe_topology += 1
          if dependency[1] * dependency[3] == 1:
            nonexist_topology += 1 

          original_nonexist_topology -= 2
          original_exist_topology -= 2
      
      if original_nonexist_topology > 0:
        for random_division_turn in range(original_nonexist_topology):
          if ran.random() < (1 - RAND_PROB) ** 2:
            nonexist_topology += 1
      if original_exist_topology > 0:
        for random_division_turn in range(original_exist_topology):
          if ran.random() < (1 - RAND_PROB) ** 2:
            probe_topology += 1

      nonexist_score, probe_score = nonexist_topology, probe_topology

      if nonexist_score < probe_score:
        score_index += 1
      elif nonexist_score == probe_score:
        score_index += 0.5
    
    if algorithm in SECOND_ORDER_METHOD:
      original_nonexist_topology = [len(out_neighbors[node1]), len(out_neighbors[node2])]
      original_exist_topology = [len(out_neighbors[node3]), len(out_neighbors[node4])]
      
      nonexist_topology = [0, 0]
      probe_topology = [0, 0]
      
      if motif == 1:
        
        for random_division_turn in range(len(out_neighbors[overlap_node]) - 1):
          if ran.random() < 1 - RAND_PROB:
            nonexist_topology[1] += 1
            probe_topology[1] += 1
        for random_division_turn in range(len(out_neighbors[other_node2]) - 1):
          if ran.random() < 1 - RAND_PROB:
            probe_topology[0] += 1
        for random_division_turn in range(len(out_neighbors[other_node1])):
          if ran.random() < 1 - RAND_PROB:
            nonexist_topology[0] += 1

      else:
        
        if motif == 2:
          if dependency[0] == 1:
            original_nonexist_topology[0] -= 1
            original_exist_topology[0] -= 1
            if ran.random() < 1 - RAND_PROB:
              probe_topology[0] += 1
              nonexist_topology[0] += 1

          if dependency[1] == 1:
            original_nonexist_topology[0] -= 1
            original_exist_topology[1] -= 1
            if ran.random() < 1 - RAND_PROB:
              nonexist_topology[0] += 1
              probe_topology[1] += 1
              
          if dependency[2] == 1:
            original_nonexist_topology[1] -= 1
            original_exist_topology[0] -= 1
            if ran.random() < 1 - RAND_PROB:
              nonexist_topology[1] += 1
              probe_topology[0] += 1
              
          if dependency[3] == 1:
            original_nonexist_topology[1] -= 1
            original_exist_topology[1] -= 1
            if ran.random() < 1 - RAND_PROB:
              nonexist_topology[1] += 1
              probe_topology[1] += 1
            
        for i, x in enumerate(original_nonexist_topology):
          if x > 0:
            for random_division_turn in range(x):
              if ran.random() < 1 - RAND_PROB:
                nonexist_topology[i] += 1
        
        for i, x in enumerate(original_exist_topology):
          if x > 1:
              for random_division_turn in range(x - 1):
                if ran.random() > RAND_PROB:
                  probe_topology[i] += 1

      if algorithm == 'PA':
        nonexist_score = nonexist_topology[0] * nonexist_topology[1]
        probe_score = probe_topology[0] * probe_topology[1]
      elif algorithm == 'SDM':
        nonexist_score = math.exp(ALPHA * probe_topology[0]) + math.exp(ALPHA * probe_topology[1])
        probe_score = math.exp(ALPHA * nonexist_topology[0]) + math.exp(ALPHA * nonexist_topology[1])

      if nonexist_score < probe_score:
        score_index += 1
      elif nonexist_score == probe_score:
        score_index += 0.5
  
    if algorithm in THIRD_ORDER_METHOD:
      new_neighbors = {}

      if motif == 1:
        three_nodes = [other_node1, overlap_node, other_node2]
        for node in three_nodes:
          new_neighbors[node] = []
          for node_node in out_neighbors[node]:
            if node_node in three_nodes:
              continue
            if ran.random() < 1 - RAND_PROB:
              new_neighbors[node].append(node_node)
          
        if other_node1 in out_neighbors[other_node2]:
          if ran.random() < 1 - RAND_PROB:
            new_neighbors[other_node1].append(other_node2)
            new_neighbors[other_node2].append(other_node1)
        
        new_cn1 = len(list(set(new_neighbors[other_node2]) & set(new_neighbors[overlap_node])))
        probe_topology = [len(new_neighbors[other_node2]) - new_cn1, len(new_neighbors[overlap_node]) - new_cn1, new_cn1]

        new_cn2 = len(list(set(new_neighbors[other_node1]) & set(new_neighbors[overlap_node])))
        nonexist_topology = [len(new_neighbors[other_node1]) - new_cn2, len(new_neighbors[overlap_node]) - new_cn2, new_cn2]

      else:

        four_nodes = [node1, node2, node3, node4]
        for node in four_nodes:
          new_neighbors[node] = []
          for node_node in out_neighbors[node]:
            if node_node in four_nodes:
              continue
            if ran.random() < 1 - RAND_PROB:
              new_neighbors[node].append(node_node)

        if motif == 2:
          if dependency[0] == 1:
            if ran.random() < 1 - RAND_PROB:
              new_neighbors[node1].append(node3)
              new_neighbors[node3].append(node1)

          if dependency[1] == 1:
            if ran.random() < 1 - RAND_PROB:
              new_neighbors[node1].append(node4)
              new_neighbors[node4].append(node1)
              
          if dependency[2] == 1:
            if ran.random() < 1 - RAND_PROB:
              new_neighbors[node2].append(node3)
              new_neighbors[node3].append(node2)
              
          if dependency[3] == 1:
            if ran.random() < 1 - RAND_PROB:
              new_neighbors[node2].append(node4)
              new_neighbors[node4].append(node2)
      
        new_cn1 = len(list(set(new_neighbors[node3]) & set(new_neighbors[node4])))
        probe_topology = [len(new_neighbors[node3]) - new_cn1, len(new_neighbors[node4]) - new_cn1, new_cn1]

        new_cn2 = len(list(set(new_neighbors[node1]) & set(new_neighbors[node2])))
        nonexist_topology = [len(new_neighbors[node1]) - new_cn2, len(new_neighbors[node2]) - new_cn2, new_cn2]
      
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
  
  
def directed_auc_theoretical_analysis(network, algorithm, RAND_PROB):
  N, M = network.number_of_nodes(), network.number_of_edges()
  all_nodes = list(network.nodes())
  all_edges = list(network.edges())

  out_neighbors = dict(zip(all_nodes, [[] for node in all_nodes]))
  for edge in network.edges():
    out_neighbors[edge[0]].append(edge[1])

  sampling_index = 0
  score_index = 0.0
  while sampling_index < THEORY_SAMPING_MAX:
    node1, node2 = ran.choice(all_nodes), ran.choice(all_nodes)
    if node1 == node2 or network.has_edge(node1, node2) or network.has_edge(node2, node1):
      continue
    sampling_index += 1
    node3, node4 = ran.choice(all_edges)

    if algorithm in FIRST_ORDER_METHOD:
      original_nonexist_topology = len(list(set(out_neighbors[node1]) & set(out_neighbors[node2])))
      original_exist_topology = len(list(set(out_neighbors[node3]) & set(out_neighbors[node4])))
      
      nonexist_topology = 0
      probe_topology = 0

      if original_nonexist_topology > 0:
        for random_division_turn in range(original_nonexist_topology):
          if ran.random() < (1 - RAND_PROB) ** 2:
            nonexist_topology += 1
      if original_exist_topology > 0:
        for random_division_turn in range(original_exist_topology):
          if ran.random() < (1 - RAND_PROB) ** 2:
            probe_topology += 1
      
      nonexist_score, probe_score = nonexist_topology, probe_topology

      if nonexist_score < probe_score:
        score_index += 1
      elif nonexist_score == probe_score:
        score_index += 0.5
    
    if algorithm in SECOND_ORDER_METHOD:
      original_nonexist_topology = [len(out_neighbors[node1]), len(out_neighbors[node2])]
      original_exist_topology = [len(out_neighbors[node3]), len(out_neighbors[node4])]
      
      nonexist_topology = [0, 0]
      probe_topology = [0, 0]

      for i, x in enumerate(original_nonexist_topology):
        if x > 0:
          for random_division_turn in range(x):
            if ran.random() > RAND_PROB:
              nonexist_topology[i] += 1
      
      for i, x in enumerate(original_exist_topology):
        if i == 0:
          if x > 1:
            for random_division_turn in range(x - 1):
              if ran.random() > RAND_PROB:
                probe_topology[i] += 1
        else:
          for random_division_turn in range(x):
            if ran.random() > RAND_PROB:
              probe_topology[i] += 1

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
      cn1 = len(list(set(out_neighbors[node3]) & set(out_neighbors[node4])))
      original_exist_topology = [
        len(out_neighbors[node3]) - cn1, 
        len(out_neighbors[node4]) - cn1,
        cn1
        ]
      cn2 = len(list(set(out_neighbors[node1]) & set(out_neighbors[node2])))
      original_nonexist_topology = [
        len(out_neighbors[node1]) - cn2, 
        len(out_neighbors[node2]) - cn2,
        cn2
        ]
      nonexist_topology = [0, 0, 0]
      probe_topology = [0, 0, 0]

      for i in range(3):
        if original_nonexist_topology[i] > 0:
          if i < 2:
            for random_division_turn in range(original_nonexist_topology[i]):
              if ran.random() > RAND_PROB:
                nonexist_topology[i] += 1
          else:
            for random_division_turn in range(original_nonexist_topology[i]):
              aa = [0, 0]
              if ran.random() < 1 - RAND_PROB:
                aa[0] = 1
              if ran.random() < 1 - RAND_PROB:
                aa[1] = 1
              
              if aa[0] + aa[1] == 2:
                nonexist_topology[2] += 1
              elif aa[0] == 1:
                nonexist_topology[0] += 1
              elif aa[1] == 1:
                nonexist_topology[1] += 1
  
        if i < 2 and original_exist_topology[i] > 1:
          for random_division_turn in range(original_exist_topology[i] - 1):
            if ran.random() > RAND_PROB:
              probe_topology[i] += 1
        elif i == 2 and original_exist_topology[i] > 0:
          for random_division_turn in range(original_exist_topology[i]):
            aa = [0, 0]
            if ran.random() < 1 - RAND_PROB:
              aa[0] = 1
            if ran.random() < 1 - RAND_PROB:
              aa[1] = 1
            
            if aa[0] + aa[1] == 2:
              probe_topology[2] += 1
            elif aa[0] == 1:
              probe_topology[0] += 1
            elif aa[1] == 1:
              probe_topology[1] += 1
    
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


def directed_auc_theoretical_analysis_combined_with_motif(network, algorithm, RAND_PROB):
  N, M = network.number_of_nodes(), network.number_of_edges()
  all_nodes = list(network.nodes())
  all_edges = list(network.edges())

  out_neighbors = dict(zip(all_nodes, [[] for node in all_nodes]))
  for edge in network.edges():
    out_neighbors[edge[0]].append(edge[1])

  sampling_index = 0
  score_index = 0.0

  while sampling_index < THEORY_SAMPING_MAX:
    node1, node2 = ran.choice(all_nodes), ran.choice(all_nodes)
    if node1 == node2 or network.has_edge(node1, node2) or network.has_edge(node2, node1):
      continue
    sampling_index += 1
    node3, node4 = ran.choice(all_edges)
    
    # dependency testing...
    
    if node1 == node3:
      motif = 1
      overlap_node = node1
      other_node1 = node2
      other_node2 = node4
    elif node1 == node4:
      motif = 1
      overlap_node = node1
      other_node1 = node2
      other_node2 = node3
    elif node2 == node3:
      motif = 1
      overlap_node = node2
      other_node1 = node1
      other_node2 = node4
    elif node2 == node4:
      motif = 1
      overlap_node = node2
      other_node1 = node1
      other_node2 = node3
    else:
      motif = 0

    if algorithm in FIRST_ORDER_METHOD:
      original_nonexist_common_neighbors = list(set(out_neighbors[node1]) & set(out_neighbors[node2]))
      original_nonexist_topology = len(original_nonexist_common_neighbors)

      original_exist_common_neighbors = list(set(out_neighbors[node3]) & set(out_neighbors[node4]))
      original_exist_topology = len(original_exist_common_neighbors)

      original_overlap = len(list(set(original_exist_common_neighbors) & set(original_nonexist_common_neighbors)))
      
      nonexist_topology = 0
      probe_topology = 0
      
      if motif == 1:
        if node3 == overlap_node:
          if node4 in original_nonexist_common_neighbors:
            original_nonexist_topology -= 1

        for random_division_turn in range(original_nonexist_topology - original_overlap):
          if ran.random() < (1 - RAND_PROB) ** 2:
            nonexist_topology += 1
        for random_division_turn in range(original_exist_topology - original_overlap):
          if ran.random() < (1 - RAND_PROB) ** 2:
            probe_topology += 1

        for random_division_turn in range(original_overlap):
          if ran.random() < 1 - RAND_PROB:
            if ran.random() < 1 - RAND_PROB:
              probe_topology += 1
            if ran.random() < 1 - RAND_PROB:
              nonexist_topology += 1
          
      else:
        if original_nonexist_topology > 0:
          for random_division_turn in range(original_nonexist_topology):
            if ran.random() < (1 - RAND_PROB) ** 2:
              nonexist_topology += 1
        if original_exist_topology > 0:
          for random_division_turn in range(original_exist_topology):
            if ran.random() < (1 - RAND_PROB) ** 2:
              probe_topology += 1
        
      nonexist_score, probe_score = nonexist_topology, probe_topology

      if nonexist_score < probe_score:
        score_index += 1
      elif nonexist_score == probe_score:
        score_index += 0.5
    
    if algorithm in SECOND_ORDER_METHOD:
      
      nonexist_topology = [0, 0]
      probe_topology = [0, 0]
      
      if motif == 1:
        original_nonexist_topology = [len(out_neighbors[other_node1]), len(out_neighbors[overlap_node])]
        original_exist_topology = [len(out_neighbors[other_node2]), len(out_neighbors[overlap_node])]

        if node3 == overlap_node:
          original_exist_topology[1] -= 1
          original_nonexist_topology[1] -= 1
        else:
          original_exist_topology[0] -= 1
        
        for random_division_turn in range(original_nonexist_topology[1]):
          if ran.random() < 1 - RAND_PROB:
            nonexist_topology[1] += 1
            probe_topology[1] += 1
        
        for random_division_turn in range(original_nonexist_topology[0]):
          if ran.random() < 1 - RAND_PROB:
            nonexist_topology[0] += 1
        
        for random_division_turn in range(original_exist_topology[0]):
          if ran.random() < 1 - RAND_PROB:
            probe_topology[0] += 1

      else:

        original_nonexist_topology = [len(out_neighbors[node1]), len(out_neighbors[node2])]
        original_exist_topology = [len(out_neighbors[node3]), len(out_neighbors[node4])]

        for i, x in enumerate(original_nonexist_topology):
          if x > 0:
            for random_division_turn in range(x):
              if ran.random() > RAND_PROB:
                nonexist_topology[i] += 1
        
        for i, x in enumerate(original_exist_topology):
          if i == 0 and x > 1:
            for random_division_turn in range(x - 1):
              if ran.random() > RAND_PROB:
                probe_topology[i] += 1
          if i == 1 and x > 0:
            for random_division_turn in range(x):
              if ran.random() > RAND_PROB:
                probe_topology[i] += 1

      if algorithm == 'PA':
        nonexist_score = nonexist_topology[0] * nonexist_topology[1]
        probe_score = probe_topology[0] * probe_topology[1]
      elif algorithm == 'SDM':
        nonexist_score = math.exp(ALPHA * probe_topology[0]) + math.exp(ALPHA * probe_topology[1])
        probe_score = math.exp(ALPHA * nonexist_topology[0]) + math.exp(ALPHA * nonexist_topology[1])

      if nonexist_score < probe_score:
        score_index += 1
      elif nonexist_score == probe_score:
        score_index += 0.5
  
    if algorithm in THIRD_ORDER_METHOD:
      new_neighbors = {}

      if motif == 1:
        three_nodes = [other_node1, overlap_node, other_node2]
        for node in three_nodes:
          new_neighbors[node] = []
          for node_node in out_neighbors[node]:
            if ran.random() < 1 - RAND_PROB:
              new_neighbors[node].append(node_node)
        
        if node4 in new_neighbors[node3]:
          new_neighbors[node3].remove(node4)
        
        new_cn1 = len(list(set(new_neighbors[other_node2]) & set(new_neighbors[overlap_node])))
        probe_topology = [len(new_neighbors[other_node2]) - new_cn1, len(new_neighbors[overlap_node]) - new_cn1, new_cn1]

        new_cn2 = len(list(set(new_neighbors[other_node1]) & set(new_neighbors[overlap_node])))
        nonexist_topology = [len(new_neighbors[other_node1]) - new_cn2, len(new_neighbors[overlap_node]) - new_cn2, new_cn2]

      else:
        four_nodes = [node1, node2, node3, node4]
        for node in four_nodes:
          new_neighbors[node] = []
          for node_node in out_neighbors[node]:
            if ran.random() < 1 - RAND_PROB:
              new_neighbors[node].append(node_node)

        new_cn1 = len(list(set(new_neighbors[node3]) & set(new_neighbors[node4])))
        probe_topology = [len(new_neighbors[node3]) - new_cn1, len(new_neighbors[node4]) - new_cn1, new_cn1]

        new_cn2 = len(list(set(new_neighbors[node1]) & set(new_neighbors[node2])))
        nonexist_topology = [len(new_neighbors[node1]) - new_cn2, len(new_neighbors[node2]) - new_cn2, new_cn2]
      
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
  # print(file_path, net_type)
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
      if line1[0] == line1[1]:
        continue
      if g.has_edge(line1[1], line1[0]) or g.has_edge(line1[0], line1[1]):
        continue
      g.add_edge(line1[0], line1[1])
  if net_type == 'undirected':
    return g.subgraph(max(nx.connected_components(g), key=len))
  elif net_type == 'directed':
    return g.subgraph(max(nx.weakly_connected_components(g), key=len))


def output(file_name, net_type):
  if 'ar' not in net_type:
    G = read_network(file_name, net_type)
  else:
    G = read_network(file_name, net_type[:-11])
  # rewire(G, minimum_rewire_turn=20, rewire_type=specific_type)
  # print('generating...')
  N, M = G.number_of_nodes(), G.number_of_edges()
  CC = round(nx.average_clustering(G),4)
  try:
    r = round(nx.degree_assortativity_coefficient(G),4)
  except:
    r = 2

  classical_simulations = []
  novel_theorys = []
  advanced_theorys = []
  # print(file_name, N, M, CC, r)
  for method1 in METHOD:
    for s in range(1, 10):
      rand_number = s * 0.1
      classical_simulations.append(str(round(link_prediction_system(G, 'rand', method1, rand_number),4)))
      if 'un' in net_type:
        novel_theorys.append(str(round(undirected_auc_theoretical_analysis(G, method1, rand_number),4)))
        advanced_theorys.append(str(round(undirected_auc_theoretical_analysis_combined_with_motif(G, method1, rand_number),4)))
      else:
        novel_theorys.append(str(round(directed_auc_theoretical_analysis(G, method1, rand_number),4)))
        advanced_theorys.append(str(round(directed_auc_theoretical_analysis_combined_with_motif(G, method1, rand_number),4)))
  
  del G
  gc.collect()

  if net_type == 'undirected':
    file_name = file_name[len(undirected_document)+1:]
  elif net_type == 'directed':
    file_name = file_name[len(directed_document)+1:]
  elif net_type == 'undirected_artificial':
    file_name = file_name[len(undirected_artificial)+1:]
  elif net_type == 'directed_artificial':
    file_name = file_name[len(directed_artificial)+1:]
  
  print('topology:', file_name, N, M, CC, r, sep='\t')
  print('old:\t', '\t'.join(classical_simulations))
  print('new:\t', '\t'.join(novel_theorys))
  print('advance:\t', '\t'.join(advanced_theorys))

  if 'un' in net_type:
    with open(undirected_result, 'a') as f:
      f.write('\t'.join([file_name, str(N), str(M), str(CC), str(r)] + classical_simulations + novel_theorys + advanced_theorys))
      f.write('\n')
  else:
    with open(directed_result, 'a') as f:
      f.write('\t'.join([file_name, str(N), str(M), str(CC), str(r)] + classical_simulations + novel_theorys + advanced_theorys))
      f.write('\n')


def undirected_dependency_test(network):
  all_nodes = list(network.nodes())
  all_edges = list(network.edges())

  sampling_index = 0
  dependency_count_1 = 0
  dependency_count_2 = 0

  while sampling_index < THEORY_SAMPING_MAX:
    node1, node2 = ran.choice(all_nodes), ran.choice(all_nodes)
    if node1 == node2 or network.has_edge(node1, node2):
      continue
    sampling_index += 1
    node3, node4 = ran.choice(all_edges)
    
    if node1 == node3:
      dependency_count_1 += 1
    elif node1 == node4:
      dependency_count_1 += 1
    elif node2 == node3:
      dependency_count_1 += 1
    elif node2 == node4:
      dependency_count_1 += 1
    else:
      if network.has_edge(node1, node3):
        dependency_count_2 += 1
      elif network.has_edge(node1, node4):
        dependency_count_2 += 1
      elif network.has_edge(node2, node3):
        dependency_count_2 += 1
      elif network.has_edge(node2, node4):
        dependency_count_2 += 1
  
  return dependency_count_1 / THEORY_SAMPING_MAX, dependency_count_2 / THEORY_SAMPING_MAX


def directed_dependency_test(network):
  all_nodes = list(network.nodes())
  all_edges = list(network.edges())

  sampling_index = 0
  dependency_count_1 = 0
  dependency_count_2 = 0

  while sampling_index < THEORY_SAMPING_MAX:
    node1, node2 = ran.choice(all_nodes), ran.choice(all_nodes)
    if node1 == node2 or network.has_edge(node1, node2) or network.has_edge(node2, node1):
      continue
    sampling_index += 1
    node3, node4 = ran.choice(all_edges)
    
    if node1 == node3:
      dependency_count_1 += 1
    elif node1 == node4:
      if network.out_degree(node4) > 0:
        dependency_count_2 += 1
    elif node2 == node3:
      dependency_count_1 += 1
    elif node2 == node4:
      if network.out_degree(node4) > 0:
        dependency_count_2 += 1
  
  return dependency_count_1 / THEORY_SAMPING_MAX, dependency_count_2 / THEORY_SAMPING_MAX


def dependency_output(file_name, net_type):
  if 'ar' not in net_type:
    G = read_network(file_name, net_type)
  else:
    G = read_network(file_name, net_type[:-11])
  
  if net_type == 'undirected':
    file_name = file_name[len(undirected_document)+1:]
  elif net_type == 'directed':
    file_name = file_name[len(directed_document)+1:]
  
  if 'un' in net_type:
    m1, m2 = undirected_dependency_test(G)
    print(file_name, m1, m2, sep='\t')
    with open(undirected_dependency_result, 'a') as f:
      f.write('\t'.join([file_name, str(m1), str(m2)]) + '\n')
  else:
    m1, m2 = directed_dependency_test(G)
    print(file_name, m1, m2, sep='\t')
    with open(directed_dependency_result, 'a') as f:
      f.write('\t'.join([file_name, str(m1), str(m2)]) + '\n')


def rand_ER(g1, connected):
  if nx.is_directed(g1):
    g = nx.DiGraph(list(g1.edges()))
  else:
    g = nx.Graph(list(g1.edges()))
  rewiring_index = 0
  error_index = 0
  if not nx.is_directed(g):
    while rewiring_index < rewiring_num and error_index < ERROR_MAX:
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
  else:
    while rewiring_index < rewiring_num and error_index < ERROR_MAX:
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
    while rewiring_index < rewiring_num and error_index < ERROR_MAX:
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
  else:
    while rewiring_index < rewiring_num and error_index < ERROR_MAX:
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
  
  return g
  

def model_rand_deg(g1):
  if nx.is_directed(g1):
    n, m = g1.number_of_nodes(), g1.number_of_edges()
    out_degree_seq = dict(g1.out_degree()).values()
    in_degree_seq = [0 for i in range(n)]
    for i in range(m):
      s = ran.randint(0, n-1)
      in_degree_seq[s] += 1

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
  if nx.is_directed(g):
    for node in g.nodes():
      d = g.degree(node)
      if d not in out_deg2nodes:
        out_deg2nodes[d] = []
      out_deg2nodes[d].append(node)
    while rewiring_index < rewiring_num and error_index < ERROR_MAX:
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
  else:
    for node in g.nodes():
      d = g.out_degree(node)
      if d not in out_deg2nodes:
        out_deg2nodes[d] = []
      out_deg2nodes[d].append(node)
    while rewiring_index < rewiring_num and error_index < ERROR_MAX:
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
  return g


def model_rand_deg_deg(g1):
  # sum1 = 0
  out_deg2nodes = {}
  node2out_deg = {}
  degdeg2num = {}

  if nx.is_directed(g1):
    g = nx.DiGraph()
    g.add_nodes_from(g1.nodes())
    for node in g1.nodes():
      d = g1.out_degree(node)
      if d not in out_deg2nodes:
        out_deg2nodes[d] = []
      out_deg2nodes[d].append(node)
      node2out_deg[node] = d
      degdeg2num[d] = {}

    for edge in g1.edges():
      d1, d2 = g1.out_degree(edge[0]), g1.out_degree(edge[1])
      if d2 not in degdeg2num[d1]:
        degdeg2num[d1][d2] = 1
      else:
        degdeg2num[d1][d2] += 1

    all_deg = list(out_deg2nodes.keys())
    for d1 in all_deg:
      for d2 in all_deg:
        if d2 not in degdeg2num[d1]:
          continue
        error_turn = 0
        connected_turn = 0
        # sum1 += degdeg2num[d1][d2]
        while connected_turn < degdeg2num[d1][d2] and error_turn < 100:
          node1 = ran.choice(out_deg2nodes[d1])
          node2 = ran.choice(out_deg2nodes[d2])
          if node1 == node2 or g.has_edge(node1, node2):
            error_turn += 1
            continue
          if node2out_deg[node1] == 0:
            error_turn += 1
            continue
          g.add_edge(node1, node2)
          node2out_deg[node1] -= 1
          error_turn = 0
          connected_turn += 1
  else:
    g = nx.Graph()
    g.add_nodes_from(g1.nodes())
    for node in g1.nodes():
      d = g1.degree(node)
      if d not in out_deg2nodes:
        out_deg2nodes[d] = []
      out_deg2nodes[d].append(node)
      node2out_deg[node] = d
      degdeg2num[d] = {}

    for edge in g1.edges():
      d1, d2 = g1.degree(edge[0]), g1.degree(edge[1])
      if d2 not in degdeg2num[d1]:
        degdeg2num[d1][d2] = 1
      else:
        degdeg2num[d1][d2] += 1

    all_deg = list(out_deg2nodes.keys())
  
    for d1 in all_deg:
      for d2 in all_deg:
        if d2 not in degdeg2num[d1]:
          continue
        error_turn = 0
        connected_turn = 0
        # sum1 += degdeg2num[d1][d2]
        while connected_turn < degdeg2num[d1][d2] and error_turn < 100:
          node1 = ran.choice(out_deg2nodes[d1])
          node2 = ran.choice(out_deg2nodes[d2])
          if node1 == node2 or g.has_edge(node1, node2):
            error_turn += 1
            continue
          if node2out_deg[node1] == 0:
            out_deg2nodes[d1].remove(node1)
            error_turn += 1
            continue
          if node2out_deg[node2] == 0:
            out_deg2nodes[d2].remove(node2)
            error_turn += 1
            continue
          g.add_edge(node1, node2)
          node2out_deg[node1] -= 1
          node2out_deg[node2] -= 1
          error_turn = 0
          connected_turn += 1
    
  # print(sum1)
  return g
  
  
def theory_and_classical_aucs(module):
  if module == 0:
    # Compute two examplem datasets in parallel
    output(undirected_document + '/114.txt', 'undirected')
    output(directed_document + '/121.txt', 'directed')
  elif module == 1:
    # Compute multiple datasets in parallel
    # with open('undirected'+'_'+output_file, 'w') as f:
    #   None
    # with open('directed'+'_'+output_file, 'w') as f:
    #   None
    exist_undirected = file_read('undirected'+'_'+output_file)
    exist_directed = file_read('directed'+'_'+output_file)
    
    fs = []
    nts = []
    for file in os.listdir(undirected_document):
      if file not in exist_undirected:
        fs.append(undirected_document + '/' + file)
        nts.append('undirected')
    for file in os.listdir(directed_document):
      if file not in exist_undirected:
        fs.append(directed_document + '/' + file)
        nts.append('directed')
    parallel(output, fs, nts)
  elif module == 2:
    # compute artificial networks in parallel
    with open('undirected_artificial'+'_'+output_file, 'w') as f:
      None
    with open('directed_artificial'+'_'+output_file, 'w') as f:
      None

    fs = []
    nts = []
    for file in os.listdir(undirected_artificial):
      fs.append(undirected_artificial + '/' + file)
      nts.append('undirected_artificial')
    for file in os.listdir(directed_artificial):
      fs.append(directed_artificial + '/' + file)
      nts.append('directed_artificial')
    parallel(output, fs, nts)


def artificial_networks():
  for x in range(1, 11):
    for k in range(10):
      g = nx.barabasi_albert_graph(10000, x)
      with open(undirected_artificial+'/BA_'+str(x)+'_'+str(k)+'.txt', 'w') as f:
        for edge in g.edges():
          f.write(str(edge[0]) + '\t' + str(edge[1]) + '\n')
      
      g = nx.barabasi_albert_graph(10000, x)
      with open(directed_artificial+'/BA_'+str(x)+'_'+str(k)+'.txt', 'w') as f:
        for edge in g.edges():
          if ran.random() < 0.5:
            f.write(str(edge[0]) + '\t' + str(edge[1]) + '\n')
          else:
            f.write(str(edge[1]) + '\t' + str(edge[0]) + '\n')
      
      for p in [0.001, 0.005, 0.01, 0.05, 0.1]:
        g = nx.watts_strogatz_graph(10000, x, p)
        with open(undirected_artificial+'/WS_'+str(x)+'_'+str(k)+'.txt', 'w') as f:
          for edge in g.edges():
            f.write(str(edge[0]) + '\t' + str(edge[1]) + '\n')
        
        g = nx.watts_strogatz_graph(10000, x, p)
        with open(directed_artificial+'/WS_'+str(x)+'_'+str(k)+'.txt', 'w') as f:
          for edge in g.edges():
            if ran.random() < 0.5:
              f.write(str(edge[0]) + '\t' + str(edge[1]) + '\n')
            else:
              f.write(str(edge[0]) + '\t' + str(edge[1]) + '\n')


def rewiring_output(file_name, net_type, connected):
  rewiring = []
  
  if 'ar' in net_type:
    G = read_network(file_name, net_type[:-11])
  else:
    G = read_network(file_name, net_type)
  print(file_name, G.number_of_nodes(), G.number_of_edges(), 'begining...')
  if G.number_of_nodes() >= 100000:
    print(file_name, 'is too large!!!')
    return 0

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

  # G3 = rand_deg_deg(G, connected)
  G3 = model_rand_deg_deg(G)
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
  elif net_type == 'directed':
    file_name = file_name[len(directed_document)+1:]
  elif net_type == 'undirected_artificial':
    file_name = file_name[len(undirected_artificial)+1:]
  elif net_type == 'directed_artificial':
    file_name = file_name[len(directed_artificial)+1:]
  with open('v1_' + net_type+'_'+rewiring_file, 'a') as f:
    f.write('\t'.join([file_name] + rewiring))
    f.write('\n')


def file_read(file_name):
  nets = []
  with open(file_name, 'r') as f:
    lines = f.readlines()
    for line in lines:
      nets.append(line.strip('\n').split('\t')[0])
  return nets


def rewiring_classical_auc(module, connected=True):
  
  if module == 0:
    rewiring_output(undirected_document + '/114.txt', 'undirected', connected)
    rewiring_output(directed_document + '/121.txt', 'directed', connected)
  elif module == 1:
    with open('v1_undirected'+'_'+rewiring_file, 'w') as f:
      None
    with open('v1_directed'+'_'+rewiring_file, 'w') as f:
      None
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
  elif module == 2:
    exist_undirected = file_read('v1_undirected'+'_'+rewiring_file)
    exist_directed = file_read('v1_directed'+'_'+rewiring_file)
    fs = []
    nts = []
    cts = []
    for file in os.listdir(undirected_document):
      if file not in exist_undirected:
        fs.append(undirected_document + '/' + file)
        nts.append('undirected')
        cts.append(connected)
    for file in os.listdir(directed_document):
      if file not in exist_directed:
        fs.append(directed_document + '/' + file)
        nts.append('directed')
        cts.append(connected)
    parallel(rewiring_output, fs, nts, cts)
  elif module == 3:
    fs = []
    nts = []
    cts = []
    for file in os.listdir(undirected_artificial):
      fs.append(undirected_artificial + '/' + file)
      nts.append('undirected_artificial')
      cts.append(connected)
    for file in os.listdir(directed_artificial):
      fs.append(directed_artificial + '/' + file)
      nts.append('directed_artificial')
      cts.append(connected)
    parallel(rewiring_output, fs, nts, cts)
  

if __name__ == '__main__':
  '''
  if sys.argv[2] == '2':
    artificial_networks()
  
  if sys.argv[1] == '0':
    theory_and_classical_aucs(int(sys.argv[2]))
  elif sys.argv[1] == '1':
    rewiring_classical_auc(int(sys.argv[2]), sys.argv[3])
  '''
  
  # set output files
  undirected_result = 'v3_undirected_delta.txt'
  directed_result = 'v3_directed_delta.txt'
  undirected_dependency_result = 'v3_undirected_dependency.txt'
  directed_dependency_result = 'v3_directed_dependency.txt'

  # set file locations of datasets we used 
  document1 = './dataset/newUndirected'
  document2 = './dataset/newDirected'
  parameters = [[], []]
  for file in os.listdir(document1):
    parameters[0].append(document1 + '/' + file)
    parameters[1].append('undirected')
  for file in os.listdir(document2):
    parameters[0].append(document2 + '/' + file)
    parameters[1].append('directed')

  ## stage1: simulation versus theory in undirected and directed networks
  
  # test in one data
  # output('./dataset/newUndirected/S105_4.txt', 'undirected')
  # output('./dataset/newDirected/713039_2.txt', 'directed')
  
  # run in all data
  # with open(undirected_result, 'w') as f:
  #   None
  # with open(directed_result, 'w') as f:
  #   None
  # parallel(output, parameters[0], parameters[1])

  ## stage2: dependency test
  # test in one data
  # dependency_output('./dataset/newUndirected/test.txt', 'undirected')
  # dependency_output('./dataset/newDirected/z76.txt', 'directed')
  
  # run in all data
  with open(undirected_dependency_result, 'w') as f:
    None
  with open(directed_dependency_result, 'w') as f:
    None
  parallel(dependency_output, parameters[0], parameters[1])

  ## stage3: simulation in rewired networks
  
  # test in one data
  # rewiring_classical_auc(0)
  
  # run in all data
  # rewiring_classical_auc(1)
  
  # continue process
  # rewiring_classical_auc(2)
  
  ## stage 4: simulation in artificial networks with tunable CC and r
  
  # test in one data
  
  # run in all data
  # rewiring_classical_auc(2)
