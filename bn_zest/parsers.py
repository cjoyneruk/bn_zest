from numpy.core.fromnumeric import var
from .nodes import Node
import pandas as pd
import numpy as np
import re

def _sub_id(value):
    return re.sub('[^a-z0-9_]', '', value.lower())[:20]

def _reshape_npt(force_summation=False):

    def get_npt(row):
        npt = np.array(row['npt'])
        npt = npt.squeeze() if (len(row['shape']) == 1) else npt.reshape(row['shape'])
        
        if force_summation:
            npt = npt/npt.sum(axis=0)

        return npt

    return get_npt

def _check_npt(row):
    f = np.abs(row['npt'].sum(axis=0) - 1) > 1e-10
    if any(f.flatten()):
        raise ValueError(f"The probabilities for {row['id']} do not sum to 1. Please change or use force_summation=True when reading from file")

def _get_npt_shape(df):
    
    def get_shape(row):
        return [len(row['states']), *(len(df.loc[idx, 'states']) for idx in row['parents'])]
    
    return get_shape

def get_variables(df, force_summation=False):

    # - Number parents
    df['parents'] = df['parents'].apply(lambda x: [df['id'].to_list().index(idx) for idx in x])

    # - Get levels
    current_level = 0
    df['level'] = -1
    df.loc[df['parents'].apply(lambda x: len(x)==0), 'level'] = current_level

    # - get next level
    while sum(df['level'] == -1) > 0:
        current_level += 1
        df.loc[df['level'] == -1, 'level'] = df.loc[df['level'] == -1, 'parents'].apply(
            lambda parents: current_level if all(df.loc[parent, 'level'] >= 0 for parent in parents) else -1
        )

    # - Reshape npts
    df['shape'] = df[['states', 'parents']].apply(_get_npt_shape(df), axis=1)    
    df['npt'] = df[['npt', 'shape']].apply(_reshape_npt(force_summation=force_summation), axis=1)
    
    # - Check npt values
    if not force_summation:
        df[['id', 'npt']].apply(_check_npt, axis=1)

    # - Create column for variables
    df['variable'] = None

    prior_keys = ['id', 'name', 'states', 'description', 'npt', 'group']
    other_keys = prior_keys + ['parents']
    for level in range(0, current_level+1):
        
        

        recs = df['level']==level
        if level==0:
            df.loc[recs, 'variable'] = df.loc[recs, prior_keys].apply(lambda x: Node(**x), axis=1)
        else:
            df.loc[recs, 'parents'] = df.loc[recs, 'parents'].apply(lambda x: [df.loc[idx, 'variable'] for idx in x])            
            df.loc[recs, 'variable'] = df.loc[recs, other_keys].apply(lambda x: Node(**x), axis=1)

    return df['variable'].to_list()

def from_cmpx(data, network=0, remove_disconnected_variables=True, force_summation=False):
    model_data = data['model']['networks'][network]
    
    # - Prepare variables
    var_list = pd.json_normalize(model_data['nodes'])
    var_list = var_list.rename(columns={
        'configuration.states': 'states',
        'configuration.table.probabilities': 'npt'
    })

    if 'description' not in var_list.columns:
        var_list['description'] = None

    if 'group' not in var_list.columns:
        var_list['group'] = None

    var_list.loc[var_list['description'].isin(['', 'New Node']), 'description'] = None    
    var_list.loc[var_list['description'].isna(), 'description'] = None
    
    var_list = var_list[[ 'id', 'name','description' , 'npt', 'states']]

    var_list['id'] = var_list['id'].apply(_sub_id)

    # - prepare links to find parents
    links = pd.json_normalize(model_data['links'])
    
    for col in ['child', 'parent']:
        links[col] = links[col].apply(_sub_id)

    var_list['parents'] = var_list['id'].apply(
        lambda x: links.loc[links['child'] == x, 'parent'].to_list()
    )
    
    var_list['children'] = var_list['id'].apply(
        lambda x: links.loc[links['parent'] == x, 'child'].to_list()
    )

    if remove_disconnected_variables:
        disconnected_variables = var_list['parents'].apply(lambda x: len(x) == 0) & var_list['children'].apply(
            lambda x: len(x) == 0)
        var_list = var_list[~disconnected_variables]
    
    data = {
        'id': re.sub('[^a-z0-9_]', '', model_data['name'].lower())[:20],
        'name': model_data['name'],
        'description': None if ('description' not in model_data.keys()) else model_data['description'],
        'variables': get_variables(var_list, force_summation=force_summation)
    }

    return data

def from_dict(data, force_summation=False):

    var_list = pd.json_normalize(data['variables'])
    
    var_list['parents'] = var_list['parents'].apply(lambda x: [] if x is None else x)
    
    if 'description' not in var_list.columns:
        var_list['description'] = None

    if 'group' not in var_list.columns:
        var_list['group'] = None

    data['variables'] = get_variables(var_list, force_summation=force_summation)

    return data


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

