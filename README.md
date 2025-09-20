#  Link Prediction Theoretical Framework
We establish network predictability theory by mapping link prediction onto a spin-glass model, where network partitions correspond to spin configurations and predictability equals the system's average energy. Using the cavity method from statistical physics, we prove that global predictability decomposes into individual link contributions, enabling an efficient local sampling algorithm that reduces computational complexity from $O(M)$ to $O(M/N)$. We derive exact results for canonical network models: Erdős-Rényi networks exhibit universal predictability of 0.5 regardless of algorithm choice, establishing the random baseline, while structured networks show predictability controlled by their prior parameters. We introduce the predictability index (PI), which quantifies maximum achievable performance without information loss and accurately predicts algorithm performance under random division. Analysis of real networks validates our framework, revealing how degree heterogeneity and structural patterns govern predictability. This physics-based approach provides both theoretical insights into link prediction limits and practical tools for assessing network reconstruction potential, with implications for applications from biological network inference to social network analysis.

## Main code and results
The real network dataset we used can be downloaded from https://icon.colorado.edu/#!/networks. 
We only provide two example networks here.

```
python main.py
```
