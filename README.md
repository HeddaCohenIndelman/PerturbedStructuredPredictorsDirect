# Learning Randomly Perturbed Structured Predictors for Direct Loss Minimization
This repository is the official bipartite matchings experiment implementation of paper "Learning Randomly Perturbed Structured Predictors for Direct Loss Minimization".
In this work we learn the variance as well as the mean of randomized structured predictors and show that it balances better between the learned score function and the randomized noise. 

## Architecture
![Bi-Partite Matching Learning Architecture Diagram](BipartiteMatchingLearningDiagram.jpg?raw=true "Bi-Partite Matching Learning Architecture Diagram")

The expectancy over Gumbel noise of the loss is derived w.r.t. the parameters w of the signal and w.r.t. the parameters v of the variance
controller σ directly. The network μ has a first fully connected layer that links the sets of samples to an intermediate representation (with 32 neurons), and a second (fully connected) layer that turns those representations into batches of latent permutation matrices of dimension d by d each. The network σ has a single layer connecting input sample sequences to a single output which is then activated by a softplus activation. We have chosen such an activation to enforce a positive sigma value.


## Requirements
To install requirements:

pip install -r requirements.txt

## How to run this code
Settings to consider:

'n_numbers' controls the sequence length (d).

'batch_size' controls the number of sequences used in training.

'test_set_size' controls the number of sequences to evaluate in the test set.

Hyper-parameters to consider:

'samples_per_num_train' controls how many perturbations will be conducted for each permutation representation. We explored one or five in our experiments. Five are usually more beneficial as the sequence length increases. The results in the paper refer to five noise perturbations for each permutation representation.

A test set will be evaluated on the trained model, and the following metrics will be reported to log file:

1. Prop. wrong: the proportion of errors in sorting.

2. Prop. any wrong: the proportion of sequences where there was at least one error.



