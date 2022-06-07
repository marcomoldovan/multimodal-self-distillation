import numpy as np


def mean_reciprocal_rank(rs):
  """Score is reciprocal of the rank of the first relevant item
  First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
  Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
  >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
  >>> mean_reciprocal_rank(rs)
  0.61111111111111105
  >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
  >>> mean_reciprocal_rank(rs)
  0.5
  >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
  >>> mean_reciprocal_rank(rs)
  0.75
  Args:
      rs: Iterator of relevance scores (list or numpy) in rank order
          (first element is the first item)
  Returns:
      Mean reciprocal rank
  """
  rs = (np.asarray(r).nonzero()[0] for r in rs)
  return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def reciprocal_ranks(pairwise_similarity_results):
  rs = []
  for i, result in enumerate(pairwise_similarity_results):
    rs.append([])
    for entry in result:
      if entry['corpus_id'] == i:
        rs[i].append(1)
      else:
        rs[i].append(0)
  return rs