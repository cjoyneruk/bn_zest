import bnconstructer as bnc
import numpy as np

states = [f'{x}%' for x in np.linspace(10, 100, 10, dtype=int)]
X = bnc.Node('Input node', states=states)


npt = np.linspace(0.1, 1, 10)
npt = np.stack([1-npt, npt])
Y = bnc.Node('Output node', states=['No', 'Yes'], parents=[X], npt=npt)

model = bnc.BayesianNetwork(name='Test BN', nodes=[X, Y])

model.to_file('test_bn.cmpx')
