
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import kendalltau
import torch
#https://github.com/src-d/lapjv
from lapjv import lapjv


def my_sample_uniform_and_order(n_lists, n_numbers, prob_inc, min_range, max_range):
    """Samples uniform random numbers, return sorted lists and the indices of their original values

    Returns a 2-D tensor of n_lists lists of n_numbers sorted numbers in the [0,1]
    interval, each of them having n_numbers elements.
    Lists are increasing with probability prob_inc.
    It does so by first sampling uniform random numbers, and then sorting them.
    Therefore, sorted numbers follow the distribution of the order statistics of
    a uniform distribution.
    It also returns the random numbers and the lists of permutations p such
    p(sorted) = random.
    Notice that if one ones to build sorted numbers in different intervals, one
    might just want to re-scaled this canonical form.

    Args:
    n_lists: An int,the number of lists to be sorted.
    n_numbers: An int, the number of elements in the permutation.
    prob_inc: A float, the probability that a list of numbers will be sorted in
    increasing order.

    Returns:
    ordered: a 2-D float tensor with shape = [n_list, n_numbers] of sorted lists
     of numbers.
    random: a 2-D float tensor with shape = [n_list, n_numbers] of uniform random
     numbers.
    permutations: a 2-D int tensor with shape = [n_list, n_numbers], row i
     satisfies ordered[i, permutations[i]) = random[i,:].

    """
    # sample n_lists samples from Bernoulli with probability of prob_inc
    random =(torch.empty(n_lists, n_numbers).uniform_(min_range, max_range))
    random =random.type(torch.float32)

    ordered, permutations = torch.sort(random, descending=True)

    return (ordered, random, permutations)

def my_sample_permutations(n_permutations, n_objects):
    """Samples a batch permutations from the uniform distribution.

    Returns a sample of n_permutations permutations of n_objects indices.
    Permutations are assumed to be represented as lists of integers
    (see 'listperm2matperm' and 'matperm2listperm' for conversion to alternative
    matricial representation). It does so by sampling from a continuous
    distribution and then ranking the elements. By symmetry, the resulting
    distribution over permutations must be uniform.

    Args:
    n_permutations: An int, the number of permutations to sample.
    n_objects: An int, the number of elements in the permutation.
      the embedding sources.

    Returns:
    A 2D integer tensor with shape [n_permutations, n_objects], where each
      row is a permutation of range(n_objects)

    """
    random_pre_perm = torch.empty(n_permutations, n_objects).uniform_(0, 1)
    _, permutations = torch.topk(random_pre_perm, k = n_objects)
    return permutations

def my_permute_batch_split(batch_split, permutations):
    """Scrambles a batch of objects according to permutations.

    It takes a 3D tensor [batch_size, n_objects, object_size]
    and permutes items in axis=1 according to the 2D integer tensor
    permutations, (with shape [batch_size, n_objects]) a list of permutations
    expressed as lists. For many dimensional-objects (e.g. images), objects have
    to be flattened so they will respect the 3D format, i.e. tf.reshape(
    batch_split, [batch_size, n_objects, -1])

    Args:
    batch_split: 3D tensor with shape = [batch_size, n_objects, object_size] of
      splitted objects
    permutations: a 2D integer tensor with shape = [batch_size, n_objects] of
      permutations, so that permutations[n] is a permutation of range(n_objects)

    Returns:
    A 3D tensor perm_batch_split with the same shape as batch_split,
      so that perm_batch_split[n, j,:] = batch_split[n, perm[n,j],:]

    """
    batch_size= permutations.size()[0]
    n_objects = permutations.size()[1]

    permutations = permutations.view(batch_size, n_objects, -1)
    perm_batch_split = torch.gather(batch_split, 1, permutations)
    return perm_batch_split


def my_listperm2matperm(listperm):
    """Converts a batch of permutations to its matricial form.

    Args:
    listperm: 2D tensor of permutations of shape [batch_size, n_objects] so that
      listperm[n] is a permutation of range(n_objects).

    Returns:
    a 3D tensor of permutations matperm of
      shape = [batch_size, n_objects, n_objects] so that matperm[n, :, :] is a
      permutation of the identity matrix, with matperm[n, i, listperm[n,i]] = 1
    """
    n_objects = listperm.size()[1]
    eye = np.eye(n_objects)[listperm]
    eye= torch.tensor(eye, dtype=torch.int32)
    return eye

def my_matperm2listperm(matperm):
    """Converts a batch of permutations to its enumeration (list) form.

    Args:
    matperm: a 3D tensor of permutations of
      shape = [batch_size, n_objects, n_objects] so that matperm[n, :, :] is a
      permutation of the identity matrix. If the input is 2D, it is reshaped
      to 3D with batch_size = 1.
    dtype: output_type (tf.int32, tf.int64)

    Returns:
    A 2D tensor of permutations listperm, where listperm[n,i]
    is the index of the only non-zero entry in matperm[n, i, :]
    """
    batch_size = matperm.size()[0]
    n_objects = matperm.size()[1]
    matperm = matperm.view(-1, n_objects, n_objects)

    _, argmax = torch.max(matperm, dim=2, keepdim= True)
    argmax = argmax.view(batch_size, n_objects)
    return argmax

def my_invert_listperm(listperm):
    """Inverts a batch of permutations.

    Args:
    listperm: a 2D integer tensor of permutations listperm of
      shape = [batch_size, n_objects] so that listperm[n] is a permutation of
      range(n_objects)
    Returns:
    A 2D tensor of permutations listperm, where listperm[n,i]
    is the index of the only non-zero entry in matperm[n, i, :]
    """
    return my_matperm2listperm(torch.transpose(my_listperm2matperm(listperm), 1, 2))


def hungarian_matching(matrix_batch):
  """Solves a matching problem for a batch of matrices.

  This is a wrapper for the scipy.optimize.linear_sum_assignment function. It
  solves the optimization problem max_P sum_i,j M_i,j P_i,j with P a
  permutation matrix. Notice the negative sign; the reason, the original
  function solves a minimization problem

  Args:
    matrix_batch: A 3D tensor (a batch of matrices) with
      shape = [batch_size, N, N]. If 2D, the input is reshaped to 3D with
      batch_size = 1.

  Returns:
    listperms, a 2D integer tensor of permutations with shape [batch_size, N]
      so that listperms[n, :] is the permutation of range(N) that solves the
      problem  max_P sum_i,j M_i,j P_i,j with M = matrix_batch[n, :, :].
  """

  def hungarian(x):
    if x.ndim == 2:
      x = np.reshape(x, [1, x.shape[0], x.shape[1]])
    sol = np.zeros((x.shape[0], x.shape[1]), dtype=np.int32)
    lap_sol = np.zeros((x.shape[0], x.shape[1]), dtype=np.int32)
    for i in range(x.shape[0]):
        #sol[i, :] = linear_sum_assignment(-x[i, :])[1].astype(np.int32)
        lap_sol[i, :]= lapjv(-x[i, :])[0].astype(np.int32)

    return lap_sol

  lap_listperms = hungarian(matrix_batch.detach().cpu().numpy())
  lap_listperms = torch.from_numpy(lap_listperms)
  return lap_listperms

def my_kendall_tau(batch_perm1, batch_perm2):
  """Wraps scipy.stats kendalltau function.
  Args:
    batch_perm1: A 2D tensor (a batch of matrices) with
      shape = [batch_size, N]
    batch_perm2: same as batch_perm1

  Returns:
    A list of Kendall distances between each of the elements of the batch.
  """
  def kendalltau_batch(x, y):

    if x.ndim == 1:
      x = np.reshape(x, [1, x.shape[0]])
    if y.ndim == 1:
      y = np.reshape(y, [1, y.shape[0]])
    kendall = np.zeros((x.shape[0], 1), dtype=np.float32)
    for i in range(x.shape[0]):
      kendall[i, :] = kendalltau(x[i, :], y[i, :])[0]
    return kendall

  listkendall = kendalltau_batch(batch_perm1.numpy(), batch_perm2.numpy())
  listkendall = torch.from_numpy(listkendall)
  return listkendall


