import networkx as nx
import random as ran
import numpy as np
import matplotlib.pyplot as plt

# g = nx.erdos_renyi_graph(10, 0.3, directed=True)
# print(g.number_of_edges())

# degree_seq = dict(g.out_degree()).values()
# print(sum(degree_seq))
# g1 = nx.configuration_model(degree_seq)
# print(g1.number_of_edges())


# for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#   a = 10
#   b = 8
#   for m in range(8):
#     pro = 0
#     for i in range(10000):
#       a1, b1 = 0, 0
#       for j in range(a):
#         if ran.random() < p:
#           a1 += 1
#       for k in range(b):
#         if ran.random() < p:
#           b1 += 1
#       if a1 > b1:
#         pro += 1
#       elif a1 == b1:
#         pro += 0.5
#     print(p, m, pro/10000)
#   a -= 1
#   b -= 1  


def one2one():
  a = 101
  b = 100
  ass = [170, 140, 165] 
  bs = [150] 
  all_p = [0.001 * ip for ip in range(1001)]
  turn_max = 10000
  aucs = []

  for p in all_p:
    auc = 0

    for turn in range(turn_max):
      chosen_a = np.random.binomial(ran.choice(ass), 1 - p)
      chosen_b = np.random.binomial(ran.choice(bs), 1 - p)
      if chosen_a > chosen_b:
        auc += 1
      elif chosen_a == chosen_b:
        auc += 0.5
    
    auc /= turn_max
    aucs.append(auc)

    print(p, auc)

  plt.plot(all_p, aucs)
  plt.show()


def fun2():
  a = 100
  p = 0.1
  turn_max = 100000
  aucs = []

  for c in range(100):
    auc = 0

    for turn in range(turn_max):
      chosen_a = np.random.binomial(a, 1 - p)
      chosen_b = np.random.binomial(a + c, 1 - p)
    
      if chosen_a == chosen_b:
        auc += 1

    auc /= turn_max
    aucs.append(auc)

  plt.plot(list(range(100)), aucs)
  plt.show()


def fun3():
  a = 100
  turn_max = 100000
  p = 0.2
  aucs = []
  c = 50

  for i in range(1, c + 1):
    b = a + i
    auc = 0

    for turn in range(turn_max):
      chosen_a = np.random.binomial(a, p)
      chosen_b = np.random.binomial(b, p)
    
      if chosen_a == chosen_b:
        auc += 1
      
    auc /= turn_max
    aucs.append(auc)
  
  plt.plot(list(range(1, c + 1)), [(i+1)*aucs[i] for i in range(c)])
  plt.show()


def fun4():
  a = 3
  b = 1
  all_c = list(range(2, 11))
  all_p = [0.001 * ip for ip in range(1001)]
  turn_max = 10000
  
  def auc(x1, x2):
    auc = 0
    for turn in range(turn_max):
      chosen_x1 = np.random.binomial(x1, 1 - p)
      chosen_x2 = np.random.binomial(x2, 1 - p)
      if chosen_x1 == chosen_x2:
        auc += 1
      # elif chosen_x1 == chosen_x2:
      #   auc += 0.5
    return auc / turn_max

  for ic, c in enumerate(all_c):
    delta_aucs = []
    for p in all_p:
      # auc1 = auc(a, b)
      auc2 = auc(a + c, b + c)
      delta_auc = auc2
      delta_aucs.append(delta_auc)
      # print(p, delta_auc)

    plt.subplot(3, 3, ic + 1)
    plt.plot(all_p, delta_aucs)
    plt.title("c=" + str(c))
  
  plt.show()


def fun5():
  a = 3
  b = 1
  all_c = list(range(1, 21))
  all_p = [0.1 * ip for ip in range(1, 10)]
  turn_max = 200000
  
  def auc(x1, x2):
    auc = 0
    for turn in range(turn_max):
      chosen_x1 = np.random.binomial(x1, 1 - p)
      chosen_x2 = np.random.binomial(x2, 1 - p)
      if chosen_x1 > chosen_x2:
        auc += 1
      elif chosen_x1 == chosen_x2:
        auc += 0.5
    return auc / turn_max

  for ip, p in enumerate(all_p):
    delta_aucs = []
    for ic, c in enumerate(all_c):
      # auc1 = auc(a, b)
      auc2 = auc(a + c, b + c)
      delta_auc = auc2
      delta_aucs.append(delta_auc)
      # print(p, delta_auc)

    plt.subplot(3, 3, ip + 1)
    plt.plot(all_c, delta_aucs)
    plt.title("p=" + str(p))
  
  plt.show()


if __name__ == '__main__':
  # one2one()
  # fun2()
  # fun3()
  # fun4()
  fun5()