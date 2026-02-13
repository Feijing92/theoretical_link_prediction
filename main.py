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
import gc
import pickle
from networkx.utils import powerlaw_sequence


TURN_MAX = 10
SAMPING_MAX = 10000
THEORY_SAMPING_MAX = TURN_MAX * SAMPING_MAX
ALPHA = -0.05 # just for SDM
ALL_METHODS = ['CN', 'PA', 'SDM', 'Salton', 'Jaccard', 'Sorensen', 'HPI', 'HDI', 'LHNI']
FIRST_ORDER_METHOD = ['CN']
SECOND_ORDER_METHOD = ['PA', 'SDM']
THIRD_ORDER_METHOD = ['Salton', 'Jaccard', 'Sorensen', 'HPI', 'HDI', 'LHNI']
ERROR_MAX = 100000
ALL_DEGREE_TYPE = ['power-law', 'exponential', 'normal', 'lognormal']
METHOD = FIRST_ORDER_METHOD + SECOND_ORDER_METHOD + THIRD_ORDER_METHOD
undirected_document = './dataset/Undirected'
directed_document = './dataset/Directed'
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
  if N < 100:
    index_max = SAMPING_MAX * 10
  else:
    index_max = SAMPING_MAX

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
      while sampling_time < index_max:
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
  return score_index / TURN_MAX / index_max


def link_prediction_system_on_synthenic_data(g1, division, algorithm, RAND_PROB, output_file_name, measurement='auc'):
  if nx.is_directed(g1):
    network = nx.DiGraph()
  else:
    network = nx.Graph()
  network.add_edges_from(list(g1.edges()))
  network.add_nodes_from(list(g1.nodes()))
  # parameter statistics
  N, M = network.number_of_nodes(), network.number_of_edges()
  all_nodes = list(network.nodes())
  score_index  = 0
  turn = 0
  print(max(dict(nx.degree(network)).values()))

  non_exist_links = []
  for i in range(N):
    for j in range(i):
      node1, node2 = all_nodes[i], all_nodes[j]
      if not network.has_edge(node1, node2):
        non_exist_links.append((node1, node2))

  original_degree_distribution = list(nx.degree_histogram(g1))

  # independent replicate experiments
  all_turns = 1
  # all_turns = TURN_MAX
  while turn < all_turns:

    # network data division processing
    if division == 'rand':
      training_links, probe_links = random_division(network, RAND_PROB)
    elif division == 'fixed':
      training_links, probe_links = fixed_strategy(network, M * RAND_PROB)
    
    gg = nx.Graph()
    gg.add_nodes_from(network.nodes())
    gg.add_edges_from(training_links)

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
    
    selected_links = probe_links + non_exist_links
    selected_links.sort(key=lambda edge: score(edge[0], edge[1]), reverse=True)
    threshold_link = selected_links[len(probe_links) - 1]
    threshold_score = score(threshold_link[0], threshold_link[1])
    predicted_links = []
    upper_num = 0
    undefined_links = []
    for link in selected_links:
      this_score = score(link[0], link[1])
      if this_score > threshold_score:
        upper_num += 1
        predicted_links.append(link)
      elif this_score == threshold_score:
        undefined_links.append(link)
      else:
        break
    predicted_links = predicted_links + ran.sample(undefined_links, len(probe_links) - upper_num)

    removed_degree_distribution = list(nx.degree_histogram(gg))
    gg.add_edges_from(predicted_links)
    reconstructed_degree_distribution = list(nx.degree_histogram(gg))

  degrees = [original_degree_distribution, removed_degree_distribution, reconstructed_degree_distribution]

  with open(output_file_name, 'wb') as f:
    pickle.dump(degrees, f)

  return score_index / all_turns / SAMPING_MAX


def link_prediction_system_for_three_path(g1, division, RAND_PROB, measurement='auc'):
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
      continue
    turn += 1
    
    out_neighbors = dict(zip(all_nodes, [[] for node in all_nodes]))
    for edge in training_links:
      out_neighbors[edge[0]].append(edge[1])
      if not nx.is_directed(network):
        out_neighbors[edge[1]].append(edge[0])

    def score(edge):
      s = 0
      for v1 in out_neighbors[edge[0]]:
        for v2 in out_neighbors[edge[1]]:
          if v2 in out_neighbors[v1] or v1 in out_neighbors[v2]:
            s += 1
      return s

    if measurement == 'auc':
      sampling_time = 0
      while sampling_time < SAMPING_MAX:
        node1, node2 = ran.choice(all_nodes), ran.choice(all_nodes)
        if node1 == node2 or network.has_edge(node1, node2) or network.has_edge(node2, node1):
          continue
        sampling_time += 1
        node3, node4 = ran.choice(probe_links)

        nonexist_score = score((node1, node2))
        probe_score = score((node3, node4))
        
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


def degree_strategy(network1, m, ss):
  train, probe = [], []
  all_links = [edge for edge in network1.edges()]
  all_links.sort(key=lambda edge: nx.degree(network1, edge[0]) ** ss + nx.degree(network1, edge[1]) ** ss, reverse=True)
  probe = all_links[: int(m)]
  train = all_links[int(m): ]
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
      
      score_index += first_score(nonexist_topology, probe_topology)
    
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

      score_index += second_score(algorithm, nonexist_topology, probe_topology)
  
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
      
      score_index += third_score(algorithm, nonexist_topology, probe_topology)

  return score_index / THEORY_SAMPING_MAX


def undirected_auc_theoretical_analysis_for_three_path(network, RAND_PROB):
  N, M = network.number_of_nodes(), network.number_of_edges()
  all_nodes = list(network.nodes())
  all_edges = list(network.edges())

  out_neighbors = dict(zip(all_nodes, [[] for node in all_nodes]))
  for edge in network.edges():
    out_neighbors[edge[0]].append(edge[1])
    out_neighbors[edge[1]].append(edge[0])

  sampling_index = 0
  score_index = 0.0

  def search_links(v1, v2):
    objective_links = []
    for v3 in out_neighbors[v1]:
      if v3 != v2:
        objective_links.append((v1, v3))
    
    for v4 in out_neighbors[v2]:
      if v4 != v1:
        objective_links.append((v2, v4))

    for v3 in out_neighbors[v1]:
      for v4 in out_neighbors[v2]:
        if v3 in out_neighbors[v4]:
          if (v4, v3) not in objective_links and (v3, v4) not in objective_links:
            objective_links.append((v4, v3))
        elif v4 in out_neighbors[v3]:
          if (v4, v3) not in objective_links and (v3, v4) not in objective_links:
            objective_links.append((v3, v4))
    
    return objective_links


  while sampling_index < THEORY_SAMPING_MAX:
    node1, node2 = ran.choice(all_nodes), ran.choice(all_nodes)
    if node1 == node2 or network.has_edge(node1, node2):
      continue
    sampling_index += 1
    node3, node4 = ran.choice(all_edges)

    original_nonexist_objective_links = search_links(node1, node2)
    original_probe_objective_links = search_links(node3, node4)

    nonexist_objective_links = []
    probe_objective_links = []
    dependent_objective_links = []

    for link in original_probe_objective_links:
      if link == (node3, node4) or link == (node4, node3):
        continue
      
      if link in original_nonexist_objective_links or (link[1], link[0]) in original_nonexist_objective_links:
        dependent_objective_links.append(link)
      else:
        probe_objective_links.append(link)
    
    for link in original_nonexist_objective_links:
      if link == (node3, node4) or link == (node4, node3):
        continue
      
      if link in original_probe_objective_links or (link[1], link[0]) in original_probe_objective_links:
        None
      else:
        nonexist_objective_links.append(link)
      
    new_out_neighbors = {}
    for link in nonexist_objective_links + probe_objective_links + dependent_objective_links:

      if ran.random() > RAND_PROB:
        if link[0] not in new_out_neighbors:
          new_out_neighbors[link[0]] = [link[1]]
        else:
          new_out_neighbors[link[0]].append(link[1])
        
        if link[1] not in new_out_neighbors:
          new_out_neighbors[link[1]] = [link[0]]
        else:
          new_out_neighbors[link[1]].append(link[0])
    
    nonexist_score, probe_score = 0, 0
    if node1 in new_out_neighbors and node2 in new_out_neighbors:
      for v1 in new_out_neighbors[node1]:
        for v2 in new_out_neighbors[node2]:
          if v1 in new_out_neighbors[v2]:
            nonexist_score += 1
    
    if node3 in new_out_neighbors and node4 in new_out_neighbors:
      for v1 in new_out_neighbors[node3]:
        for v2 in new_out_neighbors[node4]:
          if v1 in new_out_neighbors[v2]:
            probe_score += 1
    
    if nonexist_score < probe_score:
      score_index += 1
    elif nonexist_score == probe_score:
      score_index += 0.5

  return score_index / THEORY_SAMPING_MAX


def directed_auc_theoretical_analysis_for_three_path(network, RAND_PROB):
  N, M = network.number_of_nodes(), network.number_of_edges()
  all_nodes = list(network.nodes())
  all_edges = list(network.edges())

  out_neighbors = dict(zip(all_nodes, [[] for node in all_nodes]))
  for edge in network.edges():
    out_neighbors[edge[0]].append(edge[1])

  sampling_index = 0
  score_index = 0.0

  def search_links(v1, v2):
    objective_links = []
    for v3 in out_neighbors[v1]:
      if v3 != v2:
        objective_links.append((v1, v3))
    
    for v4 in out_neighbors[v2]:
      if v4 != v1:
        objective_links.append((v2, v4))

    for v3 in out_neighbors[v1]:
      for v4 in out_neighbors[v2]:
        if v3 in out_neighbors[v4]:
          if (v4, v3) not in objective_links:
            objective_links.append((v4, v3))
        elif v4 in out_neighbors[v3]:
          if (v3, v4) not in objective_links:
            objective_links.append((v3, v4))
    
    return objective_links

  while sampling_index < THEORY_SAMPING_MAX:
    node1, node2 = ran.choice(all_nodes), ran.choice(all_nodes)
    if node1 == node2 or network.has_edge(node1, node2) or network.has_edge(node2, node1):
      continue
    sampling_index += 1
    node3, node4 = ran.choice(all_edges)

    original_nonexist_objective_links = search_links(node1, node2)
    original_probe_objective_links = search_links(node3, node4)

    nonexist_objective_links = []
    probe_objective_links = []
    dependent_objective_links = []

    for link in original_probe_objective_links:
      if link == (node3, node4) or link == (node4, node3):
        continue
      
      if link in original_nonexist_objective_links:
        dependent_objective_links.append(link)
      else:
        probe_objective_links.append(link)
    
    for link in original_nonexist_objective_links:
      if link == (node3, node4) or link == (node4, node3):
        continue
      
      if link in original_probe_objective_links:
        None
      else:
        nonexist_objective_links.append(link)
      
    new_out_neighbors = dict(zip(all_nodes, [[] for node in all_nodes]))
    for link in nonexist_objective_links + probe_objective_links + dependent_objective_links:

      if ran.random() > RAND_PROB:
        new_out_neighbors[link[0]].append(link[1])
    
    nonexist_score, probe_score = 0, 0

    for v1 in new_out_neighbors[node1]:
      for v2 in new_out_neighbors[node2]:
        if v1 in new_out_neighbors[v2] or v2 in new_out_neighbors[v1]:
          nonexist_score += 1
    
    for v1 in new_out_neighbors[node3]:
      for v2 in new_out_neighbors[node4]:
        if v1 in new_out_neighbors[v2] or v2 in new_out_neighbors[v1]:
          probe_score += 1
    
    if nonexist_score < probe_score:
      score_index += 1
    elif nonexist_score == probe_score:
      score_index += 0.5

  return score_index / THEORY_SAMPING_MAX


def first_score(nonexist_topology, probe_topology):
  score = 0
  nonexist_score, probe_score = nonexist_topology, probe_topology
  if nonexist_score < probe_score:
    score = 1
  elif nonexist_score == probe_score:
    score = 0.5
  return score


def second_score(algorithm, nonexist_topology, probe_topology):
  score = 0

  if algorithm == 'PA':
    nonexist_score = nonexist_topology[0] * nonexist_topology[1]
    probe_score = probe_topology[0] * probe_topology[1]
  elif algorithm == 'SDM':
    nonexist_score = 1.0 / (math.exp(ALPHA * nonexist_topology[0]) + math.exp(ALPHA * nonexist_topology[1]))
    probe_score = 1.0 / (math.exp(ALPHA * probe_topology[0]) + math.exp(ALPHA * probe_topology[1]))

  if nonexist_score < probe_score:
    score = 1
  elif nonexist_score == probe_score:
    score = 0.5
  
  return score


def third_score(algorithm, nonexist_topology, probe_topology):
  score = 0

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
    score = 1
  elif nonexist_score == probe_score:
    score = 0.5
  
  return score


def undirected_auc_theoretical_analysis_combined_with_motif(network, algorithm, RAND_PROB):
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

      add_num = first_score(nonexist_topology, probe_topology)
      score_index += add_num
    
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

      add_num = second_score(algorithm, nonexist_topology, probe_topology)
      score_index += add_num
  
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
      
      add_num = third_score(algorithm, nonexist_topology, probe_topology)
      score_index += add_num

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
      
      score_index += first_score(algorithm, nonexist_topology, probe_topology)
    
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

      score_index += second_score(algorithm, nonexist_topology, probe_topology)
  
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
    
      score_index += third_score(algorithm, nonexist_topology, probe_topology)

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
  score_index_0 = 0.0
  score_index_1 = 0.0

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
        
      added_num = first_score(algorithm, nonexist_topology, probe_topology)
      
      score_index += added_num
      if motif == 0:
        score_index_0 += added_num
      elif motif == 1:
        score_index_1 += added_num
    
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

      added_num = second_score(algorithm, nonexist_topology, probe_topology)
      score_index += added_num
      if motif == 0:
        score_index_0 += added_num
      elif motif == 1:
        score_index_1 += added_num
  
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
      
      added_num = third_score(algorithm, nonexist_topology, probe_topology)

      score_index += added_num
      if motif == 0:
        score_index_0 += added_num
      elif motif == 1:
        score_index_1 += added_num

  return score_index / THEORY_SAMPING_MAX, score_index_0 / THEORY_SAMPING_MAX, score_index_1 / THEORY_SAMPING_MAX


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


def network_topology(network):
  N, M = network.number_of_nodes(), network.number_of_edges()
  cc = round(nx.average_clustering(network),4)
  try:
    r = round(nx.degree_assortativity_coefficient(network), 4)
  except:
    r = 2
  return N, M, cc, r


def network_predictability(network, net_type, theory=False):
  topology = list(network_topology(network))
  preditability_index = []
  classical_LP = []
  efficient_sampling = []
  local_sampling = []

  for method1 in METHOD:
    preditability_index.append(undirected_auc_theoretical_analysis_combined_with_motif(network, method1, 0))

  for method1 in METHOD:
    for s in range(1, 10):
      rand_number = round(s * 0.1, 1)
      classical_LP.append(str(round(link_prediction_system(network, 'rand', method1, rand_number),4)))
      if theory:
        if 'un' in net_type:
          efficient_sampling.append(str(round(undirected_auc_theoretical_analysis(network, method1, rand_number),4)))
          local_sampling.append(undirected_auc_theoretical_analysis_combined_with_motif(network, method1, rand_number))
        else:
          efficient_sampling.append(str(round(directed_auc_theoretical_analysis(network, method1, rand_number),4)))
          local_sampling.append(directed_auc_theoretical_analysis_combined_with_motif(network, method1, rand_number)[0])
  return topology + preditability_index + classical_LP + efficient_sampling + local_sampling


def output(file_name, net_type):
  G = read_network(file_name, net_type)
  if net_type == 'undirected':
    file_name = file_name[len(undirected_document) + 1:]
  elif net_type == 'directed':
    file_name = file_name[len(directed_document) + 1:]

  result = network_predictability(G, net_type)
  deg_seq = [d for n, d in G.degree()]
  joint_deg_seq = dict(zip(deg_seq, [{} for x in deg_seq]))
  # print(result)
  for u, v in G.edges():
    d1, d2 = G.degree(u), G.degree(v)
    if d1 in joint_deg_seq[d2]:
      joint_deg_seq[d2][d1] += 1
    else:
      joint_deg_seq[d2][d1] = 1
    
    if d2 in joint_deg_seq[d1]:
      joint_deg_seq[d1][d2] += 1
    else:
      joint_deg_seq[d1][d2] = 1
  
  del G
  gc.collect()

  if 'un' in net_type:
    null_results = []
    for null_zero in range(1):
      g1 = nx.gnm_random_graph(result[0], result[1])
      result1 = network_predictability(g1, net_type)
      null_results.append(network_predictability(g1, net_type))
      del g1
      gc.collect()
    result1 = [np.average([xx[ix] for xx in null_results]) for ix in range(len(null_results[0]))]
    # print(result1)

    null_results = []
    for null_first in range(1):
      try:
        # g2 = nx.random_degree_sequence_graph(deg_seq)
        g2 = nx.Graph(nx.configuration_model(deg_seq))
        null_results.append(network_predictability(g2, net_type))
        del g2
        gc.collect()
      except:
        continue
    result2 = [np.average([xx[ix] for xx in null_results]) for ix in range(len(null_results[0]))]
    # print(result2)

    null_results = []
    for null_second in range(1):
      try:
        g3 = nx.joint_degree_graph(joint_deg_seq)
        null_results.append(network_predictability(g3, net_type))
        del g3
        gc.collect()
      except:
        continue
    result3 = [np.average([xx[ix] for xx in null_results]) for ix in range(len(null_results[0]))]
    # print(result3)

  if 'un' in net_type:
    with open(undirected_result, 'a') as f:
      f.write('\t'.join([file_name] + [str(x) for x in result + result1 + result2 + result3]) + '\n')
  else:
    with open(directed_result, 'a') as f:
      f.write('\t'.join([file_name] + [str(x) for x in result + result1 + result2 + result3]) + '\n')


def power_law_network(n, k, a):
  g = nx.Graph()
  n0 = k + 2

  # initial nodes_to_be_linked
  nodes_to_be_linked = []
  for i in range(n0):
    for j in range(n0 + a - 1):
      nodes_to_be_linked.append(i)
  
  # initial network-clique
  for i in range(n0):
    for j in range(i):
      g.add_edge(i, j)
  
  for new_node in range(n0, n):
    selectd_connected_nodes = []

    turn = 0
    while turn < k:
      node1 = ran.choice(nodes_to_be_linked)
      if node1 in selectd_connected_nodes:
        continue
      selectd_connected_nodes.append(node1)
      turn += 1

    for node in selectd_connected_nodes:
      g.add_edge(new_node, node)
      nodes_to_be_linked.append(node)
    
    for i in range(k + a):
      nodes_to_be_linked.append(new_node)

  return g


def network_calculation(G, rand_method, output_file_name):
  classical_simulations = []
  novel_theorys = []
  
  for method1 in ['SDM']:
    for rand_number in [0.2]:
      classical_simulations.append(str(round(link_prediction_system_on_synthenic_data(G, rand_method, method1, rand_number, output_file_name),4)))
      novel_theorys.append(str(round(undirected_auc_theoretical_analysis(G, method1, rand_number),4)))
  
  del G
  gc.collect()

  print('old:\t', '\t'.join(classical_simulations))
  print('new:\t', '\t'.join(novel_theorys))


def output_on_synthnic_data():
  start = time.time()

  lam = 4
  # for gamma in [-0.5, 0, 1, 2, 3]:
  #   network_calculation(power_law_network(10000, lam, int(lam*gamma)), 'rand', 'degree_BA_' + str(3 + gamma) + '.txt')
  
  for rp in [0.0001, 0.001, 0.01, 0.1, 1]:
    network_calculation(nx.watts_strogatz_graph(10000, lam * 2, rp), 'rand', 'degree_WS_' + str(rp) + '.txt')
  
  end = time.time()
  print('running time:', (end-start) / 60, 'mins', sep='\t')


def output_for_three_path(file_name, net_type):
  start = time.time()
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
  theorys = []

  for s in range(1, 10):
    rand_number = s * 0.1
    classical_simulations.append(str(round(link_prediction_system_for_three_path(G, 'rand', rand_number),4)))
    if 'un' in net_type:
      theorys.append(str(round(undirected_auc_theoretical_analysis_for_three_path(G, rand_number),4)))
    else:
      theorys.append(str(round(directed_auc_theoretical_analysis_for_three_path(G, rand_number),4)))
  
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
  print('new:\t', '\t'.join(theorys))

  if 'un' in net_type:
    with open(undirected_result_for_three_path, 'a') as f:
      f.write('\t'.join([file_name, str(N), str(M), str(CC), str(r)] + classical_simulations + theorys))
      f.write('\n')
  else:
    with open(directed_result_for_three_path, 'a') as f:
      f.write('\t'.join([file_name, str(N), str(M), str(CC), str(r)] + classical_simulations + theorys))
      f.write('\n')
  end = time.time()
  print('running time:', (end-start) / 60, 'mins', sep='\t')


def undirected_dependecy_index(original_network):
  all_nodes = list(original_network.nodes())
  N = original_network.number_of_nodes()
  exist_links = list(original_network.edges())
  
  index1, index2, index3 = 0, 0, 0

  def link2link_dependency(link1, link2):
    node1, node2 = link1
    node3, node4 = link2
    d1, d2 = nx.degree(original_network, node1), nx.degree(original_network, node2)
    d3, d4 = nx.degree(original_network, node3), nx.degree(original_network, node4)
    dis = abs(d1 + d2 - 2 - d3 - d4)

    overlap_links = 0
    total_links = d1 + d2 + d3 + d4 - 2

    if node1 == node3:
      overlap_links = d1 - 1
      if original_network.has_edge(node2, node4):
        overlap_links += 1
    elif node1 == node4:
      overlap_links = d1 - 1
      if original_network.has_edge(node2, node3):
        overlap_links += 1
    elif node2 == node3:
      overlap_links = d2 - 1
      if original_network.has_edge(node1, node4):
        overlap_links += 1
    elif node2 == node4:
      overlap_links = d2 - 1
      if original_network.has_edge(node1, node3):
        overlap_links += 1

    else:
      if original_network.has_edge(node1, node3):
        overlap_links += 1
      if original_network.has_edge(node1, node4):
        overlap_links += 1
      if original_network.has_edge(node2, node3):
        overlap_links += 1
      if original_network.has_edge(node2, node4):
        overlap_links += 1
      
    total_links -= overlap_links
    if overlap_links == 0:
      result1 = 0
      result2 = 0
    else:
      result1 = 1
      result2 = overlap_links / total_links

    dis /= total_links

    return result1, result2, dis

  sample_index = 0
  total_sample = 1000

  while sample_index < total_sample:
  
    node1, node2 = ran.choice(all_nodes), ran.choice(all_nodes)
    if node1 == node2 or original_network.has_edge(node1, node2):
      continue

    non_existent_link = (node1, node2)
    exist_link = ran.choice(exist_links)
    this_result = link2link_dependency(exist_link, non_existent_link)

    index1 += this_result[0]
    index2 += this_result[1]
    index3 += this_result[2]

    sample_index += 1

  return index1 / total_sample, index2 / total_sample, index3 / total_sample


def directed_dependecy_index(original_network):
  all_nodes = list(original_network.nodes())
  N = original_network.number_of_nodes()
  exist_links = list(original_network.edges())
  
  index1, index2, index3 = 0, 0, 0

  def link2link_dependency(link1, link2):
    node1, node2 = link1
    node3, node4 = link2
    d1, d2 = original_network.out_degree(node1), original_network.out_degree(node2)
    d3, d4 = original_network.out_degree(node3), original_network.out_degree(node4)
    dis = abs(d1 + d2 - 1 - d3 - d4)

    overlap_links = 0
    total_links = d1 + d2 + d3 + d4 - 1

    if node1 == node3:
      overlap_links = d1 - 1
    elif node1 == node4:
      overlap_links = d1 - 1
    elif node2 == node3:
      overlap_links = d2
    elif node2 == node4:
      overlap_links = d2
    
    total_links -= overlap_links
    if overlap_links == 0:
      result1 = 0
      result2 = 0
    else:
      result1 = 1
      result2 = overlap_links / total_links
    
    if total_links == 0:
      dis = 0
    else:
      dis /= total_links

    return result1, result2, dis

  sample_index = 0
  total_sample = 1000

  while sample_index < total_sample:
  
    node1, node2 = ran.choice(all_nodes), ran.choice(all_nodes)
    if node1 == node2 or original_network.has_edge(node1, node2):
      continue

    non_existent_link = (node1, node2)
    exist_link = ran.choice(exist_links)
    this_result = link2link_dependency(exist_link, non_existent_link)

    index1 += this_result[0]
    index2 += this_result[1]
    index3 += this_result[2]

    sample_index += 1

  return index1 / total_sample, index2 / total_sample, index3 / total_sample


def undirected_dependency_test(network):
  all_nodes = list(network.nodes())
  all_edges = list(network.edges())

  sampling_index = 0
  dependency_count_1 = 0
  dependency_count_2 = 0
  dependency_count_3 = 0
  dependency_count_4 = 0

  while sampling_index < THEORY_SAMPING_MAX:
    node1, node2 = ran.choice(all_nodes), ran.choice(all_nodes)
    if node1 == node2 or network.has_edge(node1, node2):
      continue
    sampling_index += 1
    node3, node4 = ran.choice(all_edges)
    motif = 0
    
    if node1 == node3:
      motif = 1
    elif node1 == node4:
      motif = 1
    elif node2 == node3:
      motif = 1
    elif node2 == node4:
      motif = 1
    else:
      connect = 0
      if network.has_edge(node1, node3):
        motif = 2
        connect += 1
      if network.has_edge(node1, node4):
        motif = 2
        connect += 1
      if network.has_edge(node2, node3):
        motif = 2
        connect += 1
      if network.has_edge(node2, node4):
        motif = 2
        connect += 1
    
    if motif == 1:
      dependency_count_1 += 1
      c1 = nx.common_neighbors(network, node1, node2)
      c2 = nx.common_neighbors(network, node3, node4)
      if len(list(set(c1) & set(c2))) > 0:
        dependency_count_2 += 1

    elif motif == 2:
      dependency_count_3 += 1
      if connect > 2:
        dependency_count_4 += 1
  
  return dependency_count_1 / THEORY_SAMPING_MAX, dependency_count_2 / THEORY_SAMPING_MAX, dependency_count_3 / THEORY_SAMPING_MAX, dependency_count_4 / THEORY_SAMPING_MAX


def directed_dependency_test(network):
  all_nodes = list(network.nodes())
  all_edges = list(network.edges())

  sampling_index = 0
  dependency_count_1 = 0
  dependency_count_2 = 0
  dependency_count_3 = 0

  while sampling_index < THEORY_SAMPING_MAX:
    node1, node2 = ran.choice(all_nodes), ran.choice(all_nodes)
    if node1 == node2 or network.has_edge(node1, node2) or network.has_edge(node2, node1):
      continue
    sampling_index += 1
    node3, node4 = ran.choice(all_edges)
    motif = 0
    
    if node1 == node3:
      dependency_count_1 += 1
      motif = 1
    elif node1 == node4:
      motif = 1
      if network.out_degree(node4) > 0:
        dependency_count_2 += 1
    elif node2 == node3:
      motif = 1
      dependency_count_1 += 1
    elif node2 == node4:
      motif = 1
      if network.out_degree(node4) > 0:
        dependency_count_2 += 1
    
    if motif == 1:
      n1 = list(nx.neighbors(network, node1))
      n2 = list(nx.neighbors(network, node2))
      n3 = list(nx.neighbors(network, node3))
      n4 = list(nx.neighbors(network, node4))
      if len(list(set(n1) & set(n2) & set(n3) & set(n4))) > 0:
        dependency_count_3 += 1
  
  return dependency_count_1 / THEORY_SAMPING_MAX, dependency_count_2 / THEORY_SAMPING_MAX, dependency_count_3 / THEORY_SAMPING_MAX


def dependency_output(file_name, net_type):
  if 'ar' not in net_type:
    G = read_network(file_name, net_type)
  else:
    G = read_network(file_name, net_type[:-11])
  
  if net_type == 'undirected':
    file_name = file_name[len(undirected_document)+1:]
  elif net_type == 'directed':
    file_name = file_name[len(directed_document)+1:]
  
  N = G.number_of_nodes()
  M = G.number_of_edges()
  aucs = []
  
  if 'un' in net_type:
    m1, m2, m3 = undirected_dependecy_index(G)
    print(file_name, m1, m2, m3, sep='\t')
    for method1 in ALL_METHODS:
      aucs.append(undirected_auc_theoretical_analysis_combined_with_motif(G, method1, 0)[0])
    with open(undirected_dependency_result, 'a') as f:
      f.write('\t'.join([file_name, str(N), str(M), str(m1), str(m2), str(m3)] + [str(auc) for auc in aucs]) + '\n')
  else:
    m1, m2, m3 = directed_dependecy_index(G)
    print(file_name, m1, m2, m3, sep='\t')
    for method1 in ALL_METHODS:
      aucs.append(directed_auc_theoretical_analysis_combined_with_motif(G, method1, 0)[0])
    with open(directed_dependency_result, 'a') as f:
      f.write('\t'.join([file_name, str(N), str(M), str(m1), str(m2), str(m3)] + [str(auc) for auc in aucs]) + '\n')


def undirected_error(network, algorithm, RAND_PROB):
  all_nodes = list(network.nodes())
  all_edges = list(network.edges())
  sampling_index = 0
  score = 0
  advanced_score = 0

  while sampling_index < THEORY_SAMPING_MAX:
    node1, node2 = ran.choice(all_nodes), ran.choice(all_nodes)
    if node1 == node2 or network.has_edge(node1, node2):
      continue
    sampling_index += 1
    node3, node4 = ran.choice(all_edges)
    motif = 0
    
    if node1 == node3:
      motif = 1
    elif node1 == node4:
      motif = 1
    elif node2 == node3:
      motif = 1
    elif node2 == node4:
      motif = 1
    else:
      if network.has_edge(node1, node3):
        motif = 2
      elif network.has_edge(node1, node4):
        motif = 2
      elif network.has_edge(node2, node3):
        motif = 2
      elif network.has_edge(node2, node4):
        motif = 2
    
    if motif > 0:

      ### score calculation

      c1 = len(list(nx.common_neighbors(network, node1, node2)))
      d1 = nx.degree(network, node1) - c1
      d2 = nx.degree(network, node2) - c1
      c2 = len(list(nx.common_neighbors(network, node3, node4)))
      d3 = nx.degree(network, node3) - c2
      d4 = nx.degree(network, node4) - c2
        
      nonexist_cn = 0
      nonexist_d1 = 0
      nonexist_d2 = 0
      probe_cn = 0
      probe_d3 = 0
      probe_d4 = 0

      for turn in range(c1):
        if ran.random() < (1 - RAND_PROB) ** 2:
          nonexist_cn += 1
      for turn in range(d1):
        if ran.random() < 1 - RAND_PROB:
          nonexist_d1 += 1
      for turn in range(d2):
        if ran.random() < 1 - RAND_PROB:
          nonexist_d2 += 1
      
      for turn in range(c2):
        if ran.random() < (1 - RAND_PROB) ** 2:
          probe_cn += 1
      for turn in range(d3 - 1):
        if ran.random() < 1 - RAND_PROB:
          probe_d3 += 1
      for turn in range(d4 - 1):
        if ran.random() < 1 - RAND_PROB:
          probe_d4 += 1
      
      if algorithm in FIRST_ORDER_METHOD:
        score += first_score(algorithm, nonexist_cn, probe_cn)
      elif algorithm in SECOND_ORDER_METHOD:
        score += second_score(algorithm, 
        [nonexist_cn + nonexist_d1, nonexist_cn + nonexist_d2], 
        [probe_cn + probe_d3, probe_cn + probe_d4])
      elif algorithm in THIRD_ORDER_METHOD:
        score += third_score(algorithm, 
        [nonexist_d1, nonexist_d2, nonexist_cn], 
        [probe_d3, probe_d4, probe_cn])
      
      ### advanced score calculation

      new_neighbors = {}
      four_nodes = [node1, node2, node3, node4]
      adjacent_links = []
      for node in four_nodes:
        new_neighbors[node] = []
        for node11 in nx.neighbors(network, node):
          if (node, node11) not in adjacent_links and (node11, node) not in adjacent_links:
            adjacent_links.append((node, node11))
      
      if (node3, node4) in adjacent_links:
        adjacent_links.remove((node3, node4))
      else:
        adjacent_links.remove((node4, node3))
      
      for edge in adjacent_links:
        if ran.random() < 1 - RAND_PROB:
          new_neighbors[edge[0]].append(edge[1])
          if edge[1] in four_nodes:
            new_neighbors[edge[1]].append(edge[0])
      
      nonexist_cn = len(list(set(new_neighbors[node1]) & set(new_neighbors[node2])))
      nonexist_d1 = len(new_neighbors[node1]) - nonexist_cn
      nonexist_d2 = len(new_neighbors[node2]) - nonexist_cn
      probe_cn = len(list(set(new_neighbors[node3]) & set(new_neighbors[node4])))
      probe_d3 = len(new_neighbors[node3]) - probe_cn
      probe_d4 = len(new_neighbors[node4]) - probe_cn

      if algorithm in FIRST_ORDER_METHOD:
        advanced_score += first_score(algorithm, nonexist_cn, probe_cn)
      elif algorithm in SECOND_ORDER_METHOD:
        advanced_score += second_score(algorithm, 
        [nonexist_cn + nonexist_d1, nonexist_cn + nonexist_d2], 
        [probe_cn + probe_d3, probe_cn + probe_d4])
      elif algorithm in THIRD_ORDER_METHOD:
        advanced_score += third_score(algorithm, 
        [nonexist_d1, nonexist_d2, nonexist_cn], 
        [probe_d3, probe_d4, probe_cn])

  return (advanced_score - score) / THEORY_SAMPING_MAX

      
def directed_error(network, algorithm, RAND_PROB):
  all_nodes = list(network.nodes())
  all_edges = list(network.edges())
  sampling_index = 0
  score = 0
  advanced_score = 0

  while sampling_index < THEORY_SAMPING_MAX:
    node1, node2 = ran.choice(all_nodes), ran.choice(all_nodes)
    if node1 == node2 or network.has_edge(node1, node2) or network.has_edge(node2, node1):
      continue
    sampling_index += 1
    node3, node4 = ran.choice(all_edges)
    motif = 0
    
    if node1 == node3:
      motif = 1
    elif node1 == node4:
      motif = 1
    elif node2 == node3:
      motif = 1
    elif node2 == node4:
      motif = 1
    
    if motif > 0:

      ### score calculation

      c1 = len(list(set(nx.neighbors(network, node1)) & set(nx.neighbors(network, node2))))
      d1 = network.out_degree(node1) - c1
      d2 = network.out_degree(node2) - c1
      c2 = len(list(set(nx.neighbors(network, node3)) & set(nx.neighbors(network, node4))))
      d3 = network.out_degree(node3) - c2
      d4 = network.out_degree(node4) - c2
        
      nonexist_cn = 0
      nonexist_d1 = 0
      nonexist_d2 = 0
      probe_cn = 0
      probe_d3 = 0
      probe_d4 = 0

      for turn in range(c1):
        if ran.random() < (1 - RAND_PROB) ** 2:
          nonexist_cn += 1
      for turn in range(d1):
        if ran.random() < 1 - RAND_PROB:
          nonexist_d1 += 1
      for turn in range(d2):
        if ran.random() < 1 - RAND_PROB:
          nonexist_d2 += 1
      
      for turn in range(c2):
        if ran.random() < (1 - RAND_PROB) ** 2:
          probe_cn += 1
      for turn in range(d3 - 1):
        if ran.random() < 1 - RAND_PROB:
          probe_d3 += 1
      for turn in range(d4):
        if ran.random() < 1 - RAND_PROB:
          probe_d4 += 1
      
      if algorithm in FIRST_ORDER_METHOD:
        score += first_score(algorithm, nonexist_cn, probe_cn)
      elif algorithm in SECOND_ORDER_METHOD:
        score += second_score(algorithm, 
        [nonexist_cn + nonexist_d1, nonexist_cn + nonexist_d2], 
        [probe_cn + probe_d3, probe_cn + probe_d4])
      elif algorithm in THIRD_ORDER_METHOD:
        score += third_score(algorithm, 
        [nonexist_d1, nonexist_d2, nonexist_cn], 
        [probe_d3, probe_d4, probe_cn])
      
      ### advanced score calculation

      new_neighbors = {}
      four_nodes = [node1, node2, node3, node4]
      adjacent_links = []
      for node in four_nodes:
        new_neighbors[node] = []
      for node in four_nodes:
        for node11 in nx.neighbors(network, node):
          if (node, node11) not in adjacent_links:
            adjacent_links.append((node, node11))
      
      adjacent_links.remove((node3, node4))
      
      for edge in adjacent_links:
        if ran.random() < 1 - RAND_PROB:
          new_neighbors[edge[0]].append(edge[1])
      # print(new_neighbors)
      nonexist_cn = len(list(set(new_neighbors[node1]) & set(new_neighbors[node2])))
      nonexist_d1 = len(new_neighbors[node1]) - nonexist_cn
      nonexist_d2 = len(new_neighbors[node2]) - nonexist_cn
      probe_cn = len(list(set(new_neighbors[node3]) & set(new_neighbors[node4])))
      probe_d3 = len(new_neighbors[node3]) - probe_cn
      probe_d4 = len(new_neighbors[node4]) - probe_cn

      if algorithm in FIRST_ORDER_METHOD:
        advanced_score += first_score(algorithm, nonexist_cn, probe_cn)
      elif algorithm in SECOND_ORDER_METHOD:
        advanced_score += second_score(algorithm, 
        [nonexist_cn + nonexist_d1, nonexist_cn + nonexist_d2], 
        [probe_cn + probe_d3, probe_cn + probe_d4])
      elif algorithm in THIRD_ORDER_METHOD:
        advanced_score += third_score(algorithm, 
        [nonexist_d1, nonexist_d2, nonexist_cn], 
        [probe_d3, probe_d4, probe_cn])

  return (advanced_score - score) / THEORY_SAMPING_MAX


def error_output(file_name, net_type):
  start = time.time()
  if 'ar' not in net_type:
    G = read_network(file_name, net_type)
  else:
    G = read_network(file_name, net_type[:-11])
  # rewire(G, minimum_rewire_turn=20, rewire_type=specific_type)
  # print('generating...')

  errors = []
  # print(file_name, N, M, CC, r)
  for method1 in METHOD:
    for s in range(1, 10):
      rand_number = s * 0.1
      if 'un' in net_type:
        errors.append(str(round(undirected_error(G, method1, rand_number), 4)))
      else:
        errors.append(str(round(directed_error(G, method1, rand_number), 4)))
  
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
  
  print('topology:', file_name, sep='\t')
  print('error:\t', '\t'.join(errors))

  if 'un' in net_type:
    with open(undirected_error_result, 'a') as f:
      f.write('\t'.join([file_name] + errors))
      f.write('\n')
  else:
    with open(directed_error_result, 'a') as f:
      f.write('\t'.join([file_name] + errors))
      f.write('\n')
  end = time.time()
  print('running time:', (end-start) / 60, 'mins', sep='\t')


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
  

def configuration_model_output(distribution, parameters):
  n = 10000

  while True:
    if distribution == "powerlaw":
      seq = [int(round(d)) for d in powerlaw_sequence(n, parameters["exponent"])]
    elif distribution == "exponential":
      seq = [int(round(d)) for d in np.random.exponential(parameters["scale"], n)]
    elif distribution == "normal":
      seq = [int(round(d)) for d in np.random.normal(parameters["mean"], parameters["std"], n)]
      seq = [max(0, d) for d in seq]
    else:
      raise ValueError("Unsupported distribution type")
    
    if sum(seq) % 2 == 0 and nx.is_valid_degree_sequence_havel_hakimi(seq):
        break

  G = nx.configuration_model(seq)
  G = nx.Graph(G)
  G.remove_edges_from(nx.selfloop_edges(G))

  result = network_predictability(G, 'undirected')
  # print(result)
  del G
  gc.collect()
  
  file_name = '_'.join([distribution] + [str(x) for x in parameters.values()])

  with open(configuration_result, 'a') as f:
    f.write('\t'.join([file_name] + [str(x) for x in result]) + '\n')


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
  undirected_result = 'v6_undirected_delta.txt'
  directed_result = 'v6_directed_delta.txt'
  undirected_result_for_three_path = 'v5_undirected_delta_for_three_path.txt'
  directed_result_for_three_path = 'v5_directed_delta_for_three_path.txt'
  undirected_dependency_result = 'v7_undirected_dependency.txt'
  directed_dependency_result = 'v7_directed_dependency.txt'
  undirected_error_result = 'v1_undirected_error.txt'
  directed_error_result = 'v1_directed_error.txt'
  configuration_result = 'v1_configuration.txt'

  # set file locations of datasets we used 
  document1 = './dataset/newUndirected'
  document2 = './dataset/newDirected'
  parameters = [[], []]
  for file in os.listdir(document1):
    parameters[0].append(document1 + '/' + file)
    parameters[1].append('undirected')
  # for file in os.listdir(document2):
  #   parameters[0].append(document2 + '/' + file)
  #   parameters[1].append('directed')

  ## step 1: classical link prediction, efficient sampling and local sampling
  
  # test in one data
  # output('./dataset/newUndirected/123.txt', 'undirected')
  # output('./dataset/newDirected/z76.txt', 'directed')
  
  # run in all data
  with open(undirected_result, 'w') as f:
    None
  # with open(directed_result, 'w') as f:
  #   None
  parallel(output, parameters[0], parameters[1])

  ## step 2: configuration model
  distributions = []
  parameters = []

  # power-law
  for powerlaw_exponent in [2, 2.5, 3, 3.5, 4, 4.5, 5]:
    for experimental_index in range(3):
      distributions.append("powerlaw")
      parameters.append({"exponent": powerlaw_exponent, "index": experimental_index})
  
  # exponential
  for exp_scale in range(2, 21, 2):
    for experimental_index in range(3):
      distributions.append("exponential")
      parameters.append({"scale": exp_scale, "index": experimental_index})
  
  # normal
  for normal_mean in range(2, 21, 2):
    for normal_std in range(1, 6):
      for experimental_index in range(3):
        distributions.append("normal")
        parameters.append({"mean": normal_mean, "std": normal_std, "index": experimental_index})
  
  # test in one data
  # configuration_model_output("powerlaw", {"exponent": 2, "index": 0})
  # configuration_model_output("powerlaw", {"exponent": 2.2, "index": 0})
  # configuration_model_output("powerlaw", {"exponent": 2.4, "index": 0})

  # run in all data
  # with open(configuration_result, 'w') as f:
  #   None
  # parallel(configuration_model_output, distributions, parameters)

  
  















  ## step 2: dependency test

  # with open(undirected_dependency_result, 'w') as f:
  #   None
  # with open(directed_dependency_result, 'w') as f:
  #   None
  
  # test in one data
  # dependency_output('./dataset/newUndirected/test.txt', 'undirected')
  # dependency_output('./dataset/newDirected/121.txt', 'directed')
  
  # run in all data
  # parallel(dependency_output, parameters[0], parameters[1])

  ## step 3: error calculation

  # test in one data
  # error_output('./dataset/newUndirected/test.txt', 'undirected')
  # error_output('./dataset/newDirected/z76.txt', 'directed')

  # run in all data
  # with open(undirected_error_result, 'w') as f:
  #   None
  # with open(directed_error_result, 'w') as f:
  #   None
  # parallel(error_output, parameters[0], parameters[1])

  ## step 4: higher topology such as three-path

  # test in one data
  # output_for_three_path('./dataset/newUndirected/test.txt', 'undirected')
  # output_for_three_path('./dataset/newDirected/z76.txt', 'directed')

  # run in all data
  # with open(undirected_result_for_three_path, 'w') as f:
  #   None
  # with open(directed_result_for_three_path, 'w') as f:
  #   None
  # parallel(output_for_three_path, parameters[0], parameters[1])

  ### dropped
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

  # output_on_synthnic_data()

