import pandas as pd
import numpy as np 
from scipy.stats import linregress
import matplotlib.pyplot as plt


def data_read_handling(data):
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
  new_index = {}
  
  with open('v2_new_score.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
      line1 = line.strip('\n').split('\t')
      for i, x in enumerate(line1[0]):
        if x == 't':
          ix = i
      net_name = line1[0][:ix+1]
      result = [line1[0][ix+1:]] + line1[1:]
      new_index[net_name] = result

  undirected_data = data_read_handling('./v2_undirected_delta.txt')
  directed_data = data_read_handling('./v2_directed_delta.txt')

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

      if key in new_index:
        undirected_num += 1
        row = [key, 'undirected', str(int(topology[0])), str(int(topology[1])), str(round(topology[2], 4)), str(round(topology[3], 4))] + [str(delta) for delta in deltas] + new_index[key]
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

      if key in new_index:
        directed_num += 1
        row = [key, 'directed', str(int(topology[0])), str(int(topology[1])), str(round(topology[2], 4)), str(round(topology[3], 4))] + [str(delta) for delta in deltas] + new_index[key]
        f.write('\t'.join(row) + '\n')
      
      if max(deltas) > 0.05:
        print(key, topology, deltas)
  
  print('data count: ', undirected_num, directed_num)
  total_num = undirected_num + directed_num
  print(round(average_delta / total_num, 4), round(average_advanced_delta / total_num, 4))


def correlation_analysis():
  undirected_deltas = [[] for i in range(delta_num)]
  directed_deltas = [[] for i in range(delta_num)]
  all_deltas = [[] for i in range(delta_num)]

  undirected_results = [[] for i in range(12)]
  directed_results = [[] for i in range(12)]
  all_results = [[] for i in range(12)]

  with open('./table_delta.txt', 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
      line1 = line.strip('\n').split('\t')
      # print(line1)
      N, M, cc, r = [float(x) for x in line1[2: 6]]
      delta = [float(x) for x in line1[6: 6 + delta_num]]
      m1, m2, m3, m4, i1, i2 = [float(x) for x in line1[6 + delta_num:]]
      
      # kk = float(line1[4])
      if line1[5] == 'nan':
        continue
      
      if line1[1] == 'undirected':
        k = 2 * M / N
        sparsity = M / N / (N-1) / 2
        result = [N, M, cc, r, k, sparsity, m1, m2, m3, m4, i1, i2]
        for i in range(12):
          undirected_results[i].append(result[i])
          all_results[i].append(result[i])
        for i in range(delta_num):
          undirected_deltas[i].append(delta[i])
          all_deltas[i].append(delta[i])

      else:
        k = M / N
        sparsity = M / N / (N-1)
        result = [N, M, cc, r, k, sparsity, m1, m2, m3, m4, i1, i2]
        for i in range(12):
          directed_results[i].append(result[i])
          all_results[i].append(result[i])
        for i in range(delta_num):
          directed_deltas[i].append(delta[i])
          all_deltas[i].append(delta[i])
      
  rows = ['N', 'M', 'CC', 'r', 'k', 'sparsity', 'motif1', 'motif2', 'motif3', 'index1', 'index2']
  methods = ['CN', 'PA', 'SDM', 'Salton', 'Jaccard', 'Sorensen', 'HPI', 'HDI', 'LHNI']
  all_p = [round(0.1 * x, 1) for x in range(1, 10)]
  columns = [m + '_' + str(p) for m in methods for p in all_p]

  total_results = [undirected_results, directed_results, all_results]
  total_deltas = [undirected_deltas, directed_deltas, all_deltas]
  titles = ['undirected', 'directed', 'all']
  
  for ix in range(3):
    results = total_results[ix]
    deltas = total_deltas[ix]
    # for method_name in ['pearson', 'kendall', 'spearman']:
    for method_name in ['pearson']:

      # optimal_error = []
      # vanishing_error = []
      errors = [[] for i in range(9)]

      for i, x in enumerate(deltas):

        x1 = pd.Series(x)
        relationships = []
        for j, y in enumerate(results[5: 9]):
          y1 = pd.Series(y)
          c = x1.corr(y1, method=method_name)
          # slope, intercept,r_value, p_value, std_err = linregress(x1, y1)
          relationships.append(c)
        
        # if i % 9 == 2:
        #   optimal_error.append(relationships)
        #   print('optimal:', x[:20])
        # elif i % 9 == 8:
        #   vanishing_error.append(relationships)
        #   print('vanishing:', x[:20])

        errors[i % 9].append(relationships)
      
      for ip, p in enumerate(all_p):
        plt.figure(figsize=(12, 12))
        for im, m in enumerate(methods):
          plt.subplot(3, 3, im + 1)
          x = np.array(rows[5: 9])
          y = np.array(errors[ip][im])
          # print('optimal:', x, y)
          plt.bar(x, y, color = ["#4CAF50","red","hotpink","#556B2F"])
          plt.title(m)
        # plt.show()
        plt.savefig(titles[ix] + '_' + method_name + '_p=' + str(p) + '.pdf', bbox_inches='tight')

    #     print('\t'.join([columns[i]] + [str(round(x, 4)) for x in relationships]))
    # print('*' * 40)


if __name__ == '__main__':
  delta_num = 81
  error = 0.02

  table_delta()
  # correlation_analysis()
  

      
  

