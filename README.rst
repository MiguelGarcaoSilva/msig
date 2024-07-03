MSig
====

MSig is a statistical framework for evaluating the significance of motifs with arbitrary multivariate order, accommodating different variable types.

Highlights
----------

- **Pattern probability**: Estimates the probability of a motif occurring by chance using a null model.
- **Significance**: Calculates motif significance based on binomial tails to assess motif recurrence in time series data.

Installation
------------

You can install the package using pip:

.. code-block:: bash

    pip install msig

Usage
-----

Here is an example of how to use MSig:

.. code-block:: python

    from MSig import Motif, NullModel
    import numpy as np

    # Load your data
    ts1 = [1, 3, 3, 5, 5, 2, 3, 3, 5, 5, 3, 3, 5, 4, 4]
    ts2 = [4.3, 4.5, 2.6, 3.0, 3.0, 1.7, 4.9, 2.9, 3.3, 1.9, 4.9, 2.5, 3.1, 1.8, 0.3]
    ts3 = ["A", "D", "B", "D", "A", "A", "A" ,"C", "C", "B", "D", "D", "C", "A", "A"]
    ts4 = ["T", "L", "T", "Z", "Z", "T", "L", "T", "Z", "T", "L", "T", "Z", "L", "L"]
    data = np.stack([np.asarray(ts1, dtype=int), np.asarray(ts2, dtype=float), np.asarray(ts3, dtype=str), np.asarray(ts4, dtype=str)])
    m, n = data.shape  # data with shape (m=4 x n=15)

    # Create the null model
    model = NullModel(data, dtypes=[int, float, str, str], model="empirical")

    # Identify the motif of length p=3 with three matches (at indices 1, 6, and 10) spanning the first, second, and fourth variables
    # with a maximum deviation threshold of δ = 0.5.
    vars = np.array([0, 1, 3])
    motif_subsequence = data[vars, 1:4]
    motif_subsequence

    # Obtain the null probability of the motif
    motif = Motif(motif_subsequence, vars, np.array([0, 0.5, 0, 0]), n_matches=3)
    probability = motif.set_pattern_probability(model, vars_indep=True)

    # Calculate the significance of the motif
    p = len(motif_subsequence[0])  # length of the motif
    max_possible_matches = n - p + 1  # maximum number of possible matches
    pvalue = motif.set_significance(max_possible_matches, data_n_variables=m, idd_correction=False)

Authors
-------

- **Miguel G. Silva** - `Webpage <https://github.com/MiguelGarcaoSilva>`_
- **Rui Henriques** - `Webpage <https://web.ist.utl.pt/rmch>`_
- **Sara C. Madeira** - `Webpage <https://saracmadeira.wordpress.com>`_

Acknowledgements
----------------

This work was partially funded by Fundação para a Ciência e a Tecnologia (FCT) through LASIGE Research Unit, UIDB/00408/2020 (`https://doi.org/10.54499/UIDB/00408/2020`_), UIDP/00408/2020 (`https://doi.org/10.54499/UIDP/00408/2020`_), INESC-ID Pluriannual, UIDB/50021/2020 (`https://doi.org/10.54499/UIDB/50021/2020`_), and a PhD research scholarship UIBD/153086/2022 to Miguel G. Silva.

How to cite
-----------

If you use this method in your research, please cite the following paper: Paper available soon.

License
-------

This project is licensed under the MIT License - see the `LICENSE <LICENSE>`_ file for details.