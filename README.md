# bn_zest (v0.2.2)

`bn_zest` is a python package designed for quickly constructing and analysing Bayesian Networks. It is essentially a lightweight wrapper around the `pomegrante` package (you can see pomegranate's [docs](https://pomegranate.readthedocs.io/en/latest/) and [BN examples](https://github.com/jmschrei/pomegranate/blob/master/tutorials/B_Model_Tutorial_4_Bayesian_Networks.ipynb)) that provides inference, fitting and structural learning algorithms for BNs, as well as other Markov models.

The main features of the current version are simple syntax for creating and performing inference on discrete BNs as well as import/export functionality for Agena's cmpx files. For a list of (hopefully) forthcoming features see the Future development section below.

## Installation

#### Requirements

`bn_zest` requires Python 3 (I'm not sure of the lowest version but 3.6 has worked for me). It also requires `pandas` and `pomegrante`, which in turn require other packages, a full list is in the requirements.txt file.

#### Instructions

It is recommended to create a virtual environment.

I have encountered issues when installing pomegranate because (for some unkown reason) Cython it is not listed in the installation dependencies even though it is! It is therefore necessary to install a few packages beforehand, this can be done with pip (probably also conda but I've not tried)

```
pip install nose joblib scipy numpy networkx Cython
```

Once those have been installed the easiest next step (if you have git on your computer) is to issue the command

```
pip install git+https://github.research.its.qmul.ac.uk/ahw387/bn_zest --no-cache-dir
```

Alternatively you can download the source code and copy to a relevant directory and type

```
pip install <path-to-directory> --no-cache-dir
```

This will also install pandas and pomegranate. *Note that because pomegranate is written in cython it needs to be compiled, so the setup can take a few minutes to complete.*

## Quick Start

*Take a look at the examples folder for examples of code for creating networks and agena import/export*

To use `bn_zest` import the relevant aspects from the module, for example

```
from bn_zest import Node, BayesianNetwork
import numpy as np
```

Then you can create nodes

```
x = Node('X', states=['No', 'Yes'], npt=[0.2,0.8])

y_npt = np.random.rand(3,2)
y_npt = y_npt/y_npt.sum(axis=0)
y = Node('Y', states=['a', 'b', 'c'], parents=[x], npt=y_npt)
```

This creates two variables `x` and `y` with states 'Yes'/'No' and 'a'/'b'/'c' respectively.
The variable y is a child of x. To view the NPTs of the variables we can write in the console

```
>>> y.npt
```

These variables can be subsequently wrapped inside a Bayesian Network by writing

```
model = BayesianNetwork('My BN', nodes=[x,y])
```

and we can then perform inference (using `pomegranate`) by issuing, for example, the command

```
marginals = model.predict_proba(input={y: 'a'})
print(marginals)
```

## Current Issues
A couple of problems to be fixed 

* *FIX MODEL NPT REPLACEMENT* - NPTs in the model are not being properly reassigned
  
* *CHECK COMPUTATIONS* - Outputs need validating against Agena

## Future development

* *Expanded node types* - In a previous implementation I had some support for Logistic Regression Nodes, Ranked Nodes and NoisyOR nodes which I aim to implement

* *Fitting* - `pomegrante` supports fitting of the NPT parameters. I think this needs to be modified to support `pandas` dataframes as inputs.

* *scikit-learn implementation* - It should be possible to configure the `BayesianNetwork` object to work with scikit-learn as a classifier model. In that case scikit-learn's tools package can be leveraged. For example cross validation support would look something like

```
from sklearn.model_selection import cross_val_score

model = bnc.BayesianNetwork('my BN', nodes=input_nodes + output_nodes)
X = data[input_node_names]
y = data[output_node_names]
score = cross_val_score(model, X, y)

```
* *cmpx import/export refinement* - For example, node descriptions, grouping and support for Ranked Node conversion

* *BNDS interface interaction* - The ability to quickly upload, download and edit models on the Newton Server

* *Documentation and examples* - Documentation both inside and outside the code needs improving. It would also be helpful to have some python notebooks

* *Plotting* - It would be nice to have plots of the marginal probabilities as well as visualisation of the network structure

* *Improved error implementation* - Improved exception raising to stop errors and bugs creeping in as well as help debugging problems.
