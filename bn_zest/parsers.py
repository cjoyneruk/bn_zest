from .nodes import Node
import pandas as pd
import numpy as np
import re


def from_cmpx(data, network=0, remove_disconnected_nodes=True):
    model_data = data['model']['networks'][network]

    node_list = pd.json_normalize(model_data['nodes'])
    links = pd.json_normalize(model_data['links'])

    node_list['parents'] = node_list['id'].apply(
        lambda x: links.loc[links['child'] == x, 'parent'].to_list()
    )

    node_list['children'] = node_list['id'].apply(
        lambda x: links.loc[links['parent'] == x, 'child'].to_list()
    )

    if remove_disconnected_nodes:
        disconnected_nodes = node_list['parents'].apply(lambda x: len(x) == 0) & node_list['children'].apply(
            lambda x: len(x) == 0)
        node_list = node_list[~disconnected_nodes]

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

    nodes = []
    for idx, row in node_list.iterrows():

        states = row['configuration.states']

        parent_orders = [node_list.loc[parent, 'order'] for parent in row['parents']]
        parent_nodes = [nodes[order] for order in parent_orders]

        # - Configure npt
        if row['level'] == 1:
            npt = np.array(row['configuration.table.probabilities']).squeeze()
        else:
            parent_sizes = [len(parent) for parent in parent_nodes]
            npt = np.array(row['configuration.table.probabilities']).reshape([len(states)] + parent_sizes)

        node_data = {
            'name': row['name'],
            'states': states,
            'parents': parent_nodes,
            'npt': npt,
            'level': row['level']
        }

        if row['description'] not in ['', 'New Node']:
            node_data['description'] = row['description']

        nodes.append(Node(**node_data))

    return model_data['name'], nodes


def _get_label(node):

    """
    Converts a node name into a label by converting to lowercase and removing all non alphanumeric characters
    :param node:
    :return:
    """

    return re.sub(r'[^0-9a-zA-Z]+', '', node.name.lower())


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
                 'id': _get_label(node)}

    return node_data


def to_cmpx(model):

    nodes = [_get_cmpx_node(node) for node in model.nodes]

    links = [{
        'parent': _get_label(parent),
        'child': _get_label(child)
    } for parent, child in model.edges]

    network = [{'nodes': nodes,
                    'links': links,
                    'name': model.name,
                    'id': re.sub(r'[^0-9a-zA-Z]+', '', model.name.lower())}]

    settings = {'parameterLearningLogging': False, 'discreteTails': False, 'sampleSizeRanked': 5, 'convergence': 0.001,
     'simulationLogging': False, 'sampleSize': 2, 'iterations': 50, 'tolerance': 1}

    return {'model': {'settings': settings, 'networks': network}}

