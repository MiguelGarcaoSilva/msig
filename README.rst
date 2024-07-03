MSig
===========

Msig is a method designed to evaluate the statistical significance of motifs from multivariate time series data.

Highlights
--------

- **Null Model**: The method uses a null model to estimate the probability of a motif occurring by chance.
- **Significance**: The method calculates the significance of a motif by comparing the probability of the motif occurring by chance with the probability of the motif occurring in the data.

Installation
------------

You can install the package using pip:

.. code-block:: bash

    pip install msig

Usage
-----

Here are some examples of how to use DataCleaner:

.. code-block:: python

    from MSig import Motif, NullModel
    import numpy as np

    # Load your data
    ts1 = [1,3,3,5,5,2,3,3,5,5,3,3,5,4,4]
    ts2 = [4.3, 4.5, 2.6, 3.0, 3.0, 1.7, 4.9, 2.9, 3.3, 1.9, 4.9, 2.5, 3.1, 1.8, 0.3]
    ts3 = ["A","D", "B", "D", "A", "A", "A" ,"C", "C", "B", "D", "D", "C", "A", "A" ]
    ts4 = ["T", "L", "T", "Z", "Z", "T", "L", "T", "Z", "T", "L", "T", "Z", "L", "L"]
    multivar_time_series = np.stack([np.asarray(ts1, dtype=int), np.asarray(ts2, dtype=float), np.asarray(ts3, dtype=str), np.asarray(ts4, dtype=str)])

    # Create a Null Model
    model = NullModel(multivar_time_series, dtypes=[int, float, str, str],  model="empirical")

    # Obtain the Null probability of the motif 
    motif = Motif(motif_subsequence, vars, np.array([0,0.5,0,0]), 3)
    p = motif.set_pattern_probability(model, vars_indep=True)

    # Calculate the significance of the motif
    max_possible_matches = 13
    pvalue = motif.set_significance(max_possible_matches, 1, idd_correction=False)

License
-------

This project is licensed under the MIT License - see the `LICENSE <LICENSE>`_ file for details.

Authors
-------

- **Miguel G. Silva** - *Initial work* - `github.com/MiguelGarcaoSilva <https://github.com/MiguelGarcaoSilva>`_

How to cite
---------------

If you use this method in your research, please cite the following paper: TODO:



