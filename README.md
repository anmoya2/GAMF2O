# Genetic_mab
A genetic algorithm following a multi-armed bandit-based approach.

The setup.py file includes the setup for using the algorithm.

The gamf2o folder contains:
* evol_genetic.py, which includes our proposal and an example of use.
* evolutionModel.py, that includes the class from which the wrapper of the model used in our genetics should inherit.

The example folder includes an example using GAMF2O


## Instalation

### Using Git repository

Git can be easily installed on widely used operating systems such as Windows, Mac, and Linux. It is worth noting that Git comes pre-installed on the majority of Mac and Linux machines by default.

<pre> ```bash $ git clone https://github.com/anmoya2/GAMF2O ``` </pre>


<pre>      
	$ cd GAMF2O
    $ python -m pip install --upgrade pip # To update pip
    $ python -m pip install --upgrade build # To update build
    $ python -m build
    $ pip install dist/gamf2o-1.0-py3-none-any.whl

</pre>