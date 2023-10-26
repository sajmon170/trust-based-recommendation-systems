# A first principles approach to trust based recommendation systems
## Abstract
This paper is an exploration of trust-based recommender systems, dissecting various algorithms and evaluating their effectiveness and robustness across varying contexts, including adversarial environments. By investigating different information types in social networks, such as item-rating and intra-item information, and the contexts in which they are effective, we gain an understanding of how these distinct information forms combine to be constructive in preference prediction. Item-rating information was identified as a pivotal feature in determining similarity between users, significantly outperforming other information types in a collaborative filtering approach. Despite its lower mean absolute error (MAE) performance, the trust graph was revealed to formulate recommenders that are notably more robust to network adversarial attacks, as they leverage the hard-to-manipulate trust structures embedded in the network. Intra-item information, although sub-optimal in isolation, enhanced the consistency of predictions and improved lower-end performance when fused with other information forms. The paper introduces the “Weighted Average” (WA) framework, a versatile and demonstrably effective approach that enables the construction of recommendation systems around any user-to-user similarity metric. Our apex recommender, the Jaccard Item Jaccard WA, amalgamates trust-based and item-based Jaccard indexes to generate user similarities, achieving a mean MAE of 0.72 across validation data sets, and outperforming our other recommenders in all test cases. Furthermore, we navigate through the challenges of the cold start problem and propose further research into the resilience of recommender systems against various adversarial strategies, potential improvements to popular models in literature and recognizing the potential for numerous computational optimizations and enhancements.

## See the [full paper](./Report.pdf) for more details (and pretty graphs) 

# Artifacts
## ./Code
The code folder contains all the code used to generate the results in the paper. The code is written in Python 3.7 and is mainly centralised in `Analysis.ipynb`.

The code for recommendation systems can be found located in `randomWalkMC.py` for the random walk methods, `baselines.py` for the baseline RS and `recommenders.py` for everything else.

- `utils.py` contains various utility functions used throughout the codebase.
  - `create_false_node_adversary_graph` is the utility function for creating an adversarial dataset.
- `common.py` initialises the datasets, precomputing the user-user similarity mappings and other data structures to cache computation so the algorithms can run faster.
- `item_diffs.py` contains functions for computing the item rating difference metrics for the IRD and WIRD recommenders.
- `intra_item_calcs.py` contains functions for computing the intra-item information similarity metrics, including pearson correlation and intra-item jaccard.

## ./datasets
The datasets folder contains the datasets used in the paper.
Each dataset contains two pieces of information, `ratings_data` and `trust_data`.
- `ratings_data`
  - list of 'node item rating\n'
- `trust_data`
  - list of 'node_from node_to 1\n'

## Running the code
Simply executing the cells in Analysis.ipynb should provide graphs of the results from cached result data.
To run the recommender algorithms, you can comment out the cell that loads the cached results.
This will take a while to run, as it has to compute the results from scratch.

The only non std-library requirement may be `tqdm`, which can be installed with `pip install tqdm`.