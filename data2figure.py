import xlrd
import matplotlib as mpl
import math as mh
import numpy as np
from scipy.stats import norm

mpl.rcParams['axes.unicode_minus'] = False


colors = ['\\colorone', '\\colortwo', '\\colorthree', '\\colorfour', '\\colorfive', '\\colorsix', '\\colorseven', '\\coloreight', '\\colornine', '\\colorten', '\\coloreleven']
class2node_size = {
  'social': 'nodeone',
  'informational': 'nodetwo',
  'technological': 'nodethree',
  'economic': 'nodefour',
  'transportation': 'nodefive',
  'biological': 'nodesix'
  }
ALL_METHODS = [r'$\mathrm{CN}$', r'$\mathrm{PA}$', r'$\mathrm{SDM}$', r'$\mathrm{Salton}$', r'$\mathrm{Jaccard}$',r'$\mathrm{S{\o}rensen}$', r'$\mathrm{HPI}$', r'$\mathrm{HDI}$', r'$\mathrm{LHN\!-\!I}$']
labels1 = [r'$0.2$', r'$0.4$', r'$0.6$', r'$0.8$']
labels2 = [r'$0.15$', r'$0.3$', r'$0.45$', r'$0.6$']
output = 'figure2-latex.txt'


def xl2data(file):
  wb = xlrd.open_workbook(file)
  sh = wb.sheet_by_name('Sheet1')
  net2class = {}
  for i in range(1, sh.nrows):
    a, b = sh.cell(i,0).value, sh.cell(i,1).value
    if isinstance(a, float):
      a = str(int(a))
    net2class[a] = b
  return net2class


def data_read(data):
  data2result = {}
  with open(data) as f:
    lines = f.readlines()
    for line in lines:
      line1 = line.strip('\n').split('\t')
      net_name = line1[0]
      topology = [float(x) for x in line1[1: 6]]
      simulation = [float(x) for x in line1[6: 87]]
      theory = [float(x) for x in line1[87:]]
      data2result[net_name] = [topology, simulation, theory]
  return data2result


def rewiring_data_read(data):
  data2result = {}
  with open(data) as f:
    lines = f.readlines()
    for line in lines:
      line1 = line.strip('\n').split('\t')
      if '-1' in line1:
        print(line1[0])
        continue
      net_name = line1[0]
      rand1 = [float(x) for x in line1[1: 82]]
      rand2 = [float(x) for x in line1[82: 163]]
      rand3 = [float(x) for x in line1[163: 244]]
      data2result[net_name] = [rand1, rand2, rand3]
  return data2result


def CDF(input_list):
  sorted_list = sorted(input_list)
  # print(sorted_list)
  points = []
  all_num = len(input_list)
  for i, x in enumerate(sorted_list[:-1]):
    if x < sorted_list[i+1]:
      points.append((x, (i+1) / all_num))
  points.append((sorted_list[-1], 1))
  # print(points)
  return points


def CDF_plot(datas, output, rightnum, logx=False, logy=False):
  example_method_index = 2
  divisions = [0, 2, 4, 6, 8]
  division2delta = {}
  for division in divisions:
    division2delta[division] = []

  print('biased networks:')
  bias_zero = 0
  bias_unnormal = 0
  net_num = 0
  for net_name, value in datas.items():
    net_num += 1
    topology, simulation, theory = value
    delta_max = topology[-1]

    for division in divisions:
      simulation_auc = simulation[example_method_index * 9 + division]
      theory_auc = theory[example_method_index * 9 + division]
      delta = abs(simulation_auc - theory_auc)
    
      if delta <= delta_max:
        if delta_max == 0:
          division2delta[division].append(0)
          bias_zero += 1
        else:
          division2delta[division].append(delta / delta_max)
          if delta / delta_max >= 0.99:
            print(net_name, simulation_auc, theory_auc, delta_max)
      elif delta <= delta_max + 0.01:
        # division2delta[division].append(delta / (delta_max + 0.01))
        bias_unnormal += 1
      else:
        print(net_name, simulation_auc, theory_auc, delta_max)
  print(net_num)
  print(bias_zero, bias_zero / net_num / 81)
  print(bias_unnormal, bias_unnormal / net_num / 81)

  with open(output, 'a') as f:
    for i, division in enumerate(divisions):
      value = division2delta[division]
      points = CDF(value)
      f.write('\\draw[line width=1.0pt,color='+colors[i]+'] ')
      strs = []
      for x, y in points:
        if logx:
          if x < 0.01:
            continue
          else:
            new_x = mh.log10(x) / 2 + 1
        else:
          new_x = x
        
        if logy:
          if y < 0.1:
            continue
          else:
            new_y = mh.log10(y) + 1
        else:
          new_y = y
        strs.append('('+str(round(new_x+rightnum,4))+'*\\len+\\rightnum,'+str(round(new_y, 4))+'*\\len)')
      f.write('--'.join(strs) + ';\n')


def auc_versus_auc(net_class, net_data, output, rightnum):
  example_division = 0
  biass = []
  with open(output, 'a') as f:
    f.write('\\foreach \\x/\\y/\\z in \n{\n')
    strs = []
    for net_name, value in net_data.items():
      topology, simulation, theory = value
      net_name1 = net_name[:-4]
      if '_' in net_name1:
        net_name1 = net_name1.split('_')[0]
      node_size = class2node_size[net_class[net_name1]]
      example_method_index = 2

      simulation_auc = simulation[example_method_index * 9 + example_division]
      theory_auc = theory[example_method_index * 9 + example_division]
      strs.append('/'.join([str(simulation_auc), str(theory_auc), node_size]))
      biass.append(abs(simulation_auc - theory_auc))
    f.write(','.join(strs) + '}\n{\n')
    f.write('\\node[\\z] at (\\x*\\len/\\interval-\\beginvalue*\\len/\\interval+' + str(rightnum) + '*\\len, \\y*\\len/\\interval-\\beginvalue*\\len/\\interval-\\downnum) {};\n}\n')
  average_bias = round(np.average(biass), 4)
  # print('average bias between AUCs is', average_bias)
  return average_bias
      

def rewiring_auc(net_class, net_data, rewiring_data, output, rightnum):
  example_division = 0
  rand1_biass = []
  rand2_biass = []
  rand3_biass = []
  with open(output, 'a') as f:
    f.write('\\foreach \\x/\\y/\\z/\\w/\\u in \n{\n')
    strs = []
    for net_name, value in rewiring_data.items():
      if net_name not in net_data:
        continue
      
      topology, simulation, theory = net_data[net_name]
      rand1, rand2, rand3 = value
      net_name1 = net_name[:-4]
      if '_' in net_name1:
        net_name1 = net_name1.split('_')[0]
      node_size = class2node_size[net_class[net_name1]]
      example_method_index = 2

      simulation_auc = simulation[example_method_index * 9 + example_division]
      rand1_auc = rand1[example_method_index * 9 + example_division]
      rand2_auc = rand2[example_method_index * 9 + example_division]
      rand3_auc = rand3[example_method_index * 9 + example_division]
      strs.append('/'.join([str(simulation_auc), str(rand1_auc), str(rand2_auc), str(rand3_auc), node_size]))
      rand1_biass.append(abs(rand1_auc - simulation_auc))
      rand2_biass.append(abs(rand2_auc - simulation_auc))
      rand3_biass.append(abs(rand3_auc - simulation_auc))
    f.write(','.join(strs) + '\n}\n{\n')
    f.write('\\node[\\u] at (\\x*\\len/\\interval-\\beginvalue*\\len/\\interval+' + str(rightnum) + '*\\len+\\rightnum, \\y*\\len/\\interval-\\beginvalue*\\len/\\interval-\\downnum) {};')
    f.write('\\node[\\u] at (\\x*\\len/\\interval-\\beginvalue*\\len/\\interval+' + str(rightnum) + '*\\len, \\z*\\len/\\interval-\\beginvalue*\\len/\\interval-2*\\downnum) {};')
    f.write('\\node[\\u] at (\\x*\\len/\\interval-\\beginvalue*\\len/\\interval+' + str(rightnum) + '*\\len+\\rightnum, \\w*\\len/\\interval-\\beginvalue*\\len/\\interval-2*\\downnum) {};')
    f.write('\n}\n')
  average_bias_for_rand1 = round(np.average(rand1_biass), 4)
  average_bias_for_rand2 = round(np.average(rand2_biass), 4)
  average_bias_for_rand3 = round(np.average(rand3_biass), 4)
  # print('average bias in rand-ER process is', average_bias_for_rand1)
  # print('average bias in rand-deg process is', average_bias_for_rand2)
  # print('average bias in rand-deg-deg process is', average_bias_for_rand3)
  return average_bias_for_rand1, average_bias_for_rand2, average_bias_for_rand3


def data2latex():
  network_data = xl2data('./network_statistics.xlsx')
  undirected_data = data_read('./undirected_delta.txt')
  directed_data = data_read('./directed_delta.txt')
  undirected_rewiring_data = rewiring_data_read('./undirected_rewiring.txt')
  directed_rewiring_data = rewiring_data_read('./directed_rewiring.txt')
  with open(output, 'w') as f:
    None

  print('undirected:')
  # Fig.2 A1
  bias1 = auc_versus_auc(network_data, undirected_data, output, 0)
  # Fig.2 B1
  CDF_plot(undirected_data, output, 0, logx=True)
  # Fig.2 D1, E1 and F1
  bias2, bias3, bias4 = rewiring_auc(network_data, undirected_data, undirected_rewiring_data, output, 0)
  undirected_biass = [bias1, bias2, bias3, bias4]

  print('*' * 40)

  print('directed:')
  # Fig.2 A2
  bias1 = auc_versus_auc(network_data, directed_data, output, 1)
  # Fig.2 B2
  CDF_plot(directed_data, output, 1, logx=True)
  # Fig.2 D2, E2 and F2
  bias2, bias3, bias4 = rewiring_auc(network_data, directed_data, directed_rewiring_data, output, 1)
  directed_biass = [bias1, bias2, bias3, bias4]
  with open(output, 'a') as f:
    f.write('\\foreach \\x/\\y/\\z/\\w in {\n')
    strs = [
      '/'.join(['1', '0', str(undirected_biass[0]), str(directed_biass[0])]),
      '/'.join(['1', '1', str(undirected_biass[1]), str(directed_biass[1])]),
      '/'.join(['2', '0', str(undirected_biass[2]), str(directed_biass[2])]),
      '/'.join(['2', '1', str(undirected_biass[3]), str(directed_biass[3])])
    ]
    f.write(','.join(strs))
    f.write('\n}\n{\n\\node at (0.95*\\len+\\y*\\rightnum, 0.1*\\len-\\x*\\downnum)[left] { $\\mathbf{bias=\\z}$};\n\\node at (0.95*\\len+\\y*\\rightnum+\\len, 0.1*\\len-\\x*\\downnum)[left] { $\\mathbf{bias=\\w}$};\n}\n')
  

if __name__ == '__main__':
  data2latex()