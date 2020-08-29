Reinforcement Learning with Q-table

Execute: python reinforcement_learning.py <problem-dataset> <iterations>

example: python reinforcement_learning.py data10-1000a.csv 1000

Output : 

Origin: 0 Goal: 9
N: 10 Iterations: 1000 Data: data10-1000a.csv
Sparsity: 0.97 gamma: 0.65
Blocked: []
Path length: 3
Shortest path: [0, 1, 2, 9]
Time: 0.1485881805419922
Correct path

# Note: Requires networkx & matplotlib.pyplot packages to be installed
-------------------------------------------------------------------------------------------------------------------------------------
Genetic Algorithm

Execute: python genetic_algorithm.py <problem-dataset>

example: python genetic_algorithm.py data10-1000a.csv

Output : 

Iteration 0: Best so far is 6 steps for a distance of 5.000000
best_route [0 4 6 1 2 9]
Iteration 2: Best so far is 5 steps for a distance of 4.000000
best_route [0 4 5 2 9]
Iteration 5: Best so far is 4 steps for a distance of 3.000000
best_route [0 1 2 9]
============================================================================
Origin: 0 Goal: 9
N: 10 Iterations: 100 Data: data10-100a.csv
Sparsity: 0.65
Blocked: []
Path length: 3
Shortest path: [0 1 2 9]
Time: 0.5554671287536621

# Note: Requires networkx, matplotlib.pyplot & pylab packages to be installed