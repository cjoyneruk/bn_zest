from pandas.core.indexes import base
from bn_zest import BayesianNetwork
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
ASSETS_DIR = os.path.join(ROOT_DIR, 'assets')

model = BayesianNetwork.from_cmpx(os.path.join(MODELS_DIR, 'bendi_bn_2020_07_21.cmpx'))

def npt_to_html(variable, **kwargs):
    return variable.npt.to_df().to_html(**kwargs)


html_tables = (npt_to_html(variable, bold_rows=True, border=1) for variable in model.variables)

with open(os.path.join(ASSETS_DIR, 'base.html'), 'r') as html_file:

    base_string = html_file.read()

tables_string = '\n <br> \n'.join(html_tables)

base_string = base_string.replace(r'{{content}}', tables_string)

with open(os.path.join(ASSETS_DIR, 'tables.html'), 'w') as output_file:

    output_file.write(base_string)