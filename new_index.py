#!/usr/bin/python
# -*- coding: UTF-8 -*-
import networkx as nx
import random as ran
from scipy.stats import norm
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from pathos.pools import ProcessPool, ThreadPool
from tqdm import tqdm
import os
import sys
import gc


TURN_MAX = 10
SAMPING_MAX = 1000
THEORY_SAMPING_MAX = TURN_MAX * SAMPING_MAX
ALPHA = -0.05 # just for SDM
ALL_METHODS = ['CN', 'PA', 'SDM', 'Salton', 'Jaccard', 'Sorensen', 'HPI', 'HDI', 'LHNI']
FIRST_ORDER_METHOD = ['CN']
SECOND_ORDER_METHOD = ['PA', 'SDM']
THIRD_ORDER_METHOD = ['Salton', 'Jaccard', 'Sorensen', 'HPI', 'HDI', 'LHNI']
ERROR_MAX = 100000
ALL_DEGREE_TYPE = ['power-law', 'exponential', 'normal', 'lognormal']
METHOD = FIRST_ORDER_METHOD + SECOND_ORDER_METHOD + THIRD_ORDER_METHOD
undirected_document = "./dataset/newUndirected"
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


def read_network(file_path, net_type):
  # print(file_path, net_type)
  if net_type == 'undirected':
    g = nx.Graph()
    file_name = file_path[24:]
  elif net_type == 'directed':
    g = nx.DiGraph()
    file_name = file_path[22:]

  with open(file_path, encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
      if '\t' in line:
        line1 = line.strip('\n').split('\t')
      else:
        line1 = line.strip('\n').split(' ')
      if line1[0] == line1[1]:
        # print('aaa')
        continue
      if g.has_edge(line1[1], line1[0]) or g.has_edge(line1[0], line1[1]):
        continue
      g.add_edge(line1[0], line1[1])
  if net_type == 'undirected':
    g = g.subgraph(max(nx.connected_components(g), key=len))
  elif net_type == 'directed':
    g = g.subgraph(max(nx.weakly_connected_components(g), key=len))
  
  n = g.number_of_nodes()
  m = g.number_of_edges()
  
  if net_type == 'undirected':
    if m >= n * (n - 1) / 2:
      return 0
  elif net_type == 'directed':
    if m >= n * (n - 1):
      return 0
  
  all_nodes = list(g.nodes())
      
  out_neighbors = dict(zip(all_nodes, [[] for node in all_nodes]))
  for edge in g.edges():
    out_neighbors[edge[0]].append(edge[1])
    if not nx.is_directed(g):
      out_neighbors[edge[1]].append(edge[0])
  
  turn = 0
  turn_max = 2000
  motif1, motif2, motif3, motif4 = 0, 0, 0, 0
  score1, score2 = 0.0, 0.0
  error = [0 for i in range(9)]

  exist_cns = []
  nonexist_cns = []
  
  while turn < turn_max:
    a, b = ran.choice(list(g.nodes())), ran.choice(list(g.nodes()))
    if a == b or g.has_edge(a, b) or g.has_edge(b, a):
      continue
    turn += 1
    sum1, sum2 = 0.0, 0.0

    c, d = ran.choice(list(g.edges()))
    cn1 = set(out_neighbors[a]) & set(out_neighbors[b])
    da, db = len(out_neighbors[a]), len(out_neighbors[b])
    cn2 = set(out_neighbors[c]) & set(out_neighbors[d])
    dc, dd = len(out_neighbors[c]), len(out_neighbors[d])
    all_cn = len(list(cn1.union(cn2)))

    c1 = len(list(cn1))
    nonexist_cns.append(c1)
    c2 = len(list(cn2))
    auc2 = auc(c2, c1)
    exist_cns.append(c2)
    # print(a, b, c, d, c1, c2)

    if a == c or a == d or b == c or b == d:
      motif1 += 1
      
      if all_cn > 0:
        motif3 += 1
        delta = len(list(cn1 & cn2))
        
        sum1 += delta
        if a == c or a == d:
          sum2 += len(out_neighbors[a])
        else:
          sum2 += len(out_neighbors[b])
        if delta > 0:
          # auc1 = real_auc(c2 - delta, c1 - delta, delta)
          # print(a, b, c, d, c1, c2, delta)
          # print(auc2)
          # print(auc1)
          # auc2 = auc1
          None
    else:
      if nx.is_directed(g):
        continue
      
      overlap = 0
      if g.has_edge(b, d):
        overlap += 1
      if g.has_edge(a, c):
          overlap += 1
      if g.has_edge(b, c):
        overlap += 1
      if g.has_edge(a, d):
          overlap += 1
      
      if overlap >= 1:
        motif2 += 1
      
      if overlap >= 3:
        motif4 += 1
        sum1 += 1

      sum2 += overlap
    
    if all_cn > 0:
      score1 += sum1 / all_cn
    score2 += sum2 / (da+db+dc+dd)

    for i in range(9):
      error[i] += auc2[i]

  e1, e2 = np.average(exist_cns), np.average(nonexist_cns)
  var1, var2 = np.var(exist_cns), np.var(nonexist_cns)
  all_z = []
  all_predictability = []

  for p in all_p + [0.99, 0.999]:
    var_z = (e1 + e2) * p * (1 - p) + (var1 + var2) * (1 - p) ** 2
    z = (1 - p) * (e1 - e2) / math.sqrt(var_z)
    all_z.append(z)

    predictability = norm.cdf(z, loc = 0, scale = 1)
    all_predictability.append(predictability)

  scores = [motif1 / turn_max, motif2 / turn_max, motif3 / turn_max, motif4 / turn_max, score1 / turn_max, score2 / turn_max]
  with open(output_file, 'a') as f:
    f.write(file_name + '\t'.join([str(x) for x in scores]) + '\n')
  print(file_name, scores)
  print('error:', [x / turn_max for x in error])
  print(e1, e2)
  print(var1, var2)
  print(all_z)
  print(all_predictability)
  return scores


def auc(a, b):
  turn_max = 1000
  error = []

  for p in all_p:
    sum1 = 0
    for turn in range(turn_max):
      random_a = np.random.binomial(a, (1-p) ** 2)
      random_b = np.random.binomial(b, (1-p) ** 2)
      if random_a > random_b:
        sum1 += 1
      elif random_a == random_b:
        sum1 += 0.5
    error.append(sum1 / turn_max)
  return error


def real_auc(a, b, delta):
  turn_max = 1000
  error = []
  all_p = [round(i * 0.1, 1) for i in range(1, 10)]

  for p in all_p:
    sum1 = 0
    for turn in range(turn_max):
      random_a = np.random.binomial(a, (1-p) ** 2)
      random_b = np.random.binomial(b, (1-p) ** 2)
      for i in range(delta):
        if ran.random() > p:
          if ran.random() > p:
            random_a += 1
          if ran.random() > p:
            random_b += 1

      if random_a > random_b:
        sum1 += 1
      elif random_a == random_b:
        sum1 += 0.5
    error.append(sum1 / turn_max)
  return error


if __name__ == '__main__':
  all_p = [round(i * 0.1, 1) for i in range(1, 10)]

  output_file = 'v3_new_score.txt'
  with open(output_file, 'w') as f:
    None
  fs = []
  nts = []
  for file in os.listdir(undirected_document):
    fs.append(undirected_document + '/' + file)
    nts.append('undirected')
  for file in os.listdir(directed_document):
    fs.append(directed_document + '/' + file)
    nts.append('directed')
  
  read_network(undirected_document + '/' + 'z155.txt', 'undirected')
  # parallel(read_network, fs, nts)
  # read_network(undirected_document + '/' + '100.txt', 'undirected')
