import pandas as pd
import numpy as np 
from scipy.stats import linregress
import matplotlib.pyplot as plt


def topology_and_aucs_processing(data):
  data2result = {}
  with open(data) as f:
    lines = f.readlines()
    for line in lines:
      line1 = line.strip('\n').split('\t')
      net_name = line1[0]
      topology = [float(x) for x in line1[1: 5]]
      simulation = [float(x) for x in line1[5: 86]]
      theory = [float(x) for x in line1[86:167]]
      advanced_theory = [float(x) for x in line1[167:]]
      data2result[net_name] = [topology, simulation, theory, advanced_theory]
  
  return data2result


def dependency_processing(data):
  data2result = {}
  with open(data) as f:
    lines = f.readlines()
    for line in lines:
      line1 = line.strip('\n').split('\t')
      net_name = line1[0]
      data2result[net_name] = line1[1: ]
  
  return data2result


def delta_calculation(net_name, topology, simulation, theory, advanced_theory):
  deltas = []
  for i in range(delta_num):
    # abs_aucs = [round(abs(simulation[i] - theory[i]), 4) for i in range(delta_num)]
    delta = round(abs(simulation[i] - theory[i]), 4)
    deltas.append(delta)
  
  advanced_deltas = []
  for i in range(9):
    abs_aucs = [round(abs(simulation[i] - advanced_theory[i]), 4) for i in range(i*9, i*9+9)]
    delta = round(np.average(abs_aucs), 4)
    advanced_deltas.append(delta)

  if np.average(deltas) > error:
    print(net_name + ':', topology, np.average(deltas), np.average(advanced_deltas))
    # print(deltas)
    # print(advanced_deltas)
    # print('*' * 40)
  
  return deltas, np.average(deltas), np.average(advanced_deltas)


def table_delta():
  undirected_data = topology_and_aucs_processing('./v3_undirected_delta.txt')
  directed_data = topology_and_aucs_processing('./v3_directed_delta.txt')
  undirected_dependency = dependency_processing('v4_undirected_dependency.txt')
  directed_dependency = dependency_processing('v4_directed_dependency.txt')

  directed_num = 0
  undirected_num = 0
  average_delta = 0
  average_advanced_delta = 0

  with open('table_delta.txt', 'w') as f:
    f.write('\t'.join(['net_name', 'type', 'N', 'M', 'CC', 'r', 'delta_1']) + '\n')
    print('undirected data processing: ')
    for key, value in undirected_data.items():
      
      topology, simulation, theory, advanced_theory = value
      deltas, d, ad = delta_calculation(key, topology, simulation, theory, advanced_theory)
      average_delta += d
      average_advanced_delta += ad

      if key in undirected_dependency:
        undirected_num += 1
        row = [key, 'undirected', str(int(topology[0])), str(int(topology[1])), str(round(topology[2], 4)), str(round(topology[3], 4))] + [str(delta) for delta in deltas] + undirected_dependency[key]
        f.write('\t'.join(row) + '\n')

      # cn_deltas = [deltas[i] for i in range(9, 27)]
      # if max(cn_deltas) > 0.05:
      #   print(key, topology, cn_deltas)
    
    print('directed data processing: ')
    for key, value in directed_data.items():
      
      topology, simulation, theory, advanced_theory = value
      deltas, d, ad = delta_calculation(key, topology, simulation, theory, advanced_theory)
      average_delta += d
      average_advanced_delta += ad

      if key in directed_dependency:
        directed_num += 1
        row = [key, 'directed', str(int(topology[0])), str(int(topology[1])), str(round(topology[2], 4)), str(round(topology[3], 4))] + [str(delta) for delta in deltas] + directed_dependency[key]
        f.write('\t'.join(row) + '\n')
      
      # if max(deltas) > 0.05:
      #   print(key, topology, deltas)
  
  print('data count: ', undirected_num, directed_num)
  total_num = undirected_num + directed_num
  print(round(average_delta / total_num, 4), round(average_advanced_delta / total_num, 4))


def correlation_analysis():
  undirected_deltas = [[] for i in range(delta_num)]
  directed_deltas = [[] for i in range(delta_num)]

  undirected_results = [[] for i in range(5)]
  directed_results = [[] for i in range(4)]

  with open('./table_delta.txt', 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
      line1 = line.strip('\n').split('\t')
      if line1[5] == 'nan':
        continue

      N, M, cc, r = [float(x) for x in line1[2: 6]]
      delta = [float(x) for x in line1[6: 6 + delta_num]]

      if line1[1] == 'undirected':
        m1, m2, m3, m4 = [float(x) for x in line1[6 + delta_num:]]
        sparsity = M / N / (N-1) / 2
        result = [sparsity, m1, m2, m3, m4]
        for i, x in enumerate(result):
          undirected_results[i].append(x)
        for i in range(delta_num):
          undirected_deltas[i].append(delta[i])
      else:
        m1, m2, m3 = [float(x) for x in line1[6 + delta_num:]]
        sparsity = M / N / (N-1)
        result = [sparsity, m1, m2, m3]
        for i, x in enumerate(result):
          directed_results[i].append(x)
        for i in range(delta_num):
          directed_deltas[i].append(delta[i])
  
  methods = ['CN', 'PA', 'SDM', 'Salton', 'Jaccard', 'Sorensen', 'HPI', 'HDI', 'LHNI']
  all_p = [round(0.1 * x, 1) for x in range(1, 10)]
  colors = ["#4CAF50", "red", "hotpink", "#556B2F", "deepskyblue"]
  markers = ["o", "v", ">", "D", "s"]
  label_names = ['sparsity', 'm1', 'm2', 'm3', 'm4']
  corrs = []

  for i, x in enumerate(undirected_deltas):
    x1 = pd.Series(x)
    relationships = []
    for j, y in enumerate(undirected_results):
      y1 = pd.Series(y)
      c = x1.corr(y1, method='pearson')
      relationships.append(c)
    corrs.append(relationships)
      
  plt.figure(figsize=(12, 12))
  for im, m in enumerate(methods):
    plt.subplot(3, 3, im + 1)
    x = np.array(all_p)
    
    for ixx in range(5):
      y = np.array([corrs[im * 9 + ip][ixx] for ip, p in enumerate(all_p)])
      plt.plot(x, y, linestyle="-", color=colors[ixx], marker=markers[ixx], label = label_names[ixx])
    if im in [6, 7, 8]:
      plt.xlabel('p')
    if im in [0, 3, 6]:
      plt.ylabel('r')
    if im == 1:
      plt.legend()

    plt.title(m)

  plt.savefig('undirected_corr.pdf', bbox_inches='tight')

  corrs = []
  for i, x in enumerate(directed_deltas):
    x1 = pd.Series(x)
    relationships = []
    for j, y in enumerate(directed_results):
      y1 = pd.Series(y)
      c = x1.corr(y1, method='pearson')
      relationships.append(c)
    corrs.append(relationships)
      
  plt.figure(figsize=(12, 12))
  for im, m in enumerate(methods):
    plt.subplot(3, 3, im + 1)
    x = np.array(all_p)
    
    for ixx in range(4):
      y = np.array([corrs[im * 9 + ip][ixx] for ip, p in enumerate(all_p)])
      plt.plot(x, y, linestyle="-", color=colors[ixx], marker=markers[ixx], label = label_names[ixx])
    if im in [6, 7, 8]:
      plt.xlabel('p')
    if im in [0, 3, 6]:
      plt.ylabel('r')
    if im == 1:
      plt.legend()

    plt.title(m)

  plt.savefig('directed_corr.pdf', bbox_inches='tight')


if __name__ == '__main__':
  delta_num = 81
  error = 0.01

  table_delta()
  correlation_analysis()
  

      
  

