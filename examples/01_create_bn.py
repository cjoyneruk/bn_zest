from bn_zest import Node, BayesianNetwork
import numpy as np

# - Create prior nodes
X = Node('X', states=['a', 'b', 'c'], npt=[0.2, 0.3, 0.5])
Y = Node('Y', states=['low', 'medium', 'high'], npt=[0.9, 0.05, 0.05])

print('Y npt:')
print(Y.NPT)
print('')

# - Create random npt of desired shape
np.random.seed(121)
npt_Z = np.random.rand(2, 3, 3)

# - Create conditional nodes
Z = Node('Z', states=['No', 'Yes'], parents=[X, Y], npt=npt_Z)

print('Z npt:\n')
print(Z.NPT)
print('')

# - Wrap in a Bayesian Network model
model = BayesianNetwork('Sample model', nodes=[X, Y, Z])

print('Nodes:', model.nodes)
print('')

print('Edges:', model.edges)
print('')

# - Get marginals
marginals = model.predict_proba()
print(marginals['X'])

# - Get marginals with supplied evidence
marginals = model.predict_proba({Z: 'No'})
print(marginals['X'])

# - Export model to agena file
model.to_file('files/sample_bn.cmpx')