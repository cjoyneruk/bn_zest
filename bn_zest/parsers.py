from .nodes import Node
import pandas as pd
import numpy as np
import re


def from_cmpx(data, network=0, remove_disconnected_variables=True, force_summation=False):
    model_data = data['model']['networks'][network]

    node_list = pd.json_normalize(model_data['nodes'])
    links = pd.json_normalize(model_data['links'])

    node_list['parents'] = node_list['id'].apply(
        lambda x: links.loc[links['child'] == x, 'parent'].to_list()
    )

    node_list['children'] = node_list['id'].apply(
        lambda x: links.loc[links['parent'] == x, 'child'].to_list()
    )

    if remove_disconnected_variables:
        disconnected_variables = node_list['parents'].apply(lambda x: len(x) == 0) & node_list['children'].apply(
            lambda x: len(x) == 0)
        node_list = node_list[~disconnected_variables]

    node_list = node_list.set_index('id')

    # - get_levels
    current_level = 1
    node_list['level'] = 0
    node_list.loc[~node_list.index.isin(links['child']), 'level'] = current_level

    # - get next level
    while sum(node_list['level'] == 0) > 0:
        current_level += 1
        node_list.loc[node_list['level'] == 0, 'level'] = node_list.loc[node_list['level'] == 0, 'parents'].apply(
            lambda parents: current_level if all(node_list.loc[parent, 'level'] > 0 for parent in parents) else 0
        )

    node_list = node_list.sort_values(by='level')
    node_list['order'] = range(0, node_list.shape[0])

    variables = []
    for idx, row in node_list.iterrows():

        states = row['configuration.states']

        parent_orders = [node_list.loc[parent, 'order'] for parent in row['parents']]
        parent_variables = [variables[order] for order in parent_orders]

        # - Configure npt
        if row['level'] == 1:
            npt = np.array(row['configuration.table.probabilities']).squeeze()
        else:
            parent_sizes = [len(parent) for parent in parent_variables]
            npt = np.array(row['configuration.table.probabilities']).reshape([len(states)] + parent_sizes)
        
        if force_summation:
            npt = npt/npt.sum(axis=0)

        f = np.abs(npt.sum(axis=0) - 1) > 1e-10
        if any(f.flatten()):
            raise ValueError(f"The probabilities for {row['name']} do not sum to 1. Please change or use force_summation=True when reading from cmpx file")

        node_data = {
            'name': row['name'],
            'states': states,
            'npt': npt,
            'level': row['level'],
            'id': re.sub('[^a-z0-9_]', '', idx.lower())[:20]
        }


        if len(parent_variables) > 0:
            node_data['parents'] = parent_variables

        if row['description'] not in ['', 'New Node']:
            node_data['description'] = row['description']

        variables.append(Node(**node_data))

    description = None if ('description' not in model_data.keys()) else model_data['description']
    return {
        'id': re.sub('[^a-z0-9_]', '', model_data['name'].lower())[:20],
        'name': model_data['name'],
        'description': description,
        'variables': variables
    }


def _get_cmpx_node(node):

    node_type = 'Labelled'

    if len(node) == 2:
        node_type = 'Boolean'

    node_table = {'nptCompiled': True,
                  'type': 'Manual',
                  'probabilities': node.npt.to_df().values.tolist()}

    config = {'type': node_type,
              'table': node_table,
              'states': node.states}

    try:
        description = node.description
    except AttributeError:
        description = 'New node'

    node_data = {'configuration': config,
                 'name': node.name,
                 'description': description,
                 'id': node.id}

    return node_data


def to_cmpx(model):

    variables = [_get_cmpx_node(node) for node in model.variables]

    links = [{
        'parent': parent.id,
        'child': child.id
    } for parent, child in model.edges]

    network = {'nodes': variables,
                    'links': links,
                    'name': model.name,
                    'id': model.id}

    settings = {'parameterLearningLogging': False, 'discreteTails': False, 'sampleSizeRanked': 5, 'convergence': 0.001,
     'simulationLogging': False, 'sampleSize': 2, 'iterations': 50, 'tolerance': 1}

    return {'model': {'settings': settings, 'networks': [network]}}

def from_dict(data):

    node_list = pd.json_normalize(data['variables'])

    # - get_levels
    current_level = 1
    node_list['level'] = 0
    node_list.loc[node_list['parents'].isna(), 'level'] = 1

    node_list['parents'] = node_list['parents'].apply(lambda x: [] if x is None else x)
    node_list = node_list.set_index('id')

    # - get next level
    while sum(node_list['level'] == 0) > 0:
        current_level += 1
        node_list.loc[node_list['level'] == 0, 'level'] = node_list.loc[node_list['level'] == 0, 'parents'].apply(
            lambda parents: current_level if all(node_list.loc[parent, 'level'] > 0 for parent in parents) else 0
        )

    node_list = node_list.sort_values(by='level')
    node_list['order'] = range(0, node_list.shape[0])
    node_list['parents'] = node_list['parents'].apply(lambda parents: [node_list.loc[parent, 'order'] for parent in parents])

    variables = []

    for idx, row in node_list.iterrows():

        node_data = {
            'id': idx,
            'name': row['name'],
            'states': row['states'],
            'description': row['description'],
            'npt': np.array(row['npt']),        
        }

        if row['level'] == 1:
            node_data['npt'] = node_data['npt'].squeeze()
        else:
            node_data['parents'] = [variables[i] for i in row['parents']]
            shape = [len(row['states']), *(len(variable) for variable in node_data['parents'])]
            node_data['npt'] = node_data['npt'].reshape(shape)

        variables.append(Node(**node_data))

    data['variables'] = variables

    return data
