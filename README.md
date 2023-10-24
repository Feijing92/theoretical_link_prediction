#  Link Prediction Theoretical Framework
In our paper, we introduce a new theoretical analysis method for general link prediction methods, in which the proposed local link prediction can be regarded as an approximation of classical link prediction. 
Further more, we give an error computing method for the error between local link prediction and classical link prediction.
In particular, we discussed nine local-topology-based link prediction algorithms based on the degree and the common neighbor.
In these cases, theoretical analysis can be directed soved by random graph tools and/or percolaton tools (generating functions).
Error analysis method is given by the distribution of higher order topology in original networks (a simple example can be found in Fig.1).

## Calculation of AUC and error bound in two LP processes
The datasets we used are generated from https://icon.colorado.edu/#!/networks. 
We only provide two example networks here.

```
# theory_and_classical_aucs(): Calculation of AUC and error bound in two LP processes
# rewiring_classical_auc(): Calculation of AUC for classical LP in rewired networks
python main.py
```
