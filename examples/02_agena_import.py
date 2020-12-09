from bn_zest import BayesianNetwork

# - Import model from cmpx file
model = BayesianNetwork.from_file('files/sample_bn.cmpx')

# - Look at the nodes and edges
print('Nodes:', model.nodes)
print('')
print('Edges:', model.edges)
print('')

# - Get one of the nodes and print out the NPT
x = model['X']
print('X NPT:')
print(x.npt)
print('')

# - Run the model
marginals = model.predict_proba({'Z': 'Yes'})
print(marginals['X'])
print('')