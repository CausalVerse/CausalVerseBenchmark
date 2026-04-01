"""Mean Correlation Coefficient from Hyvarinen & Morioka
   Taken from https://github.com/bethgelab/slow_disentanglement/blob/master/mcc_metric/metric.py
"""
import sys
sys.path.append('../')
import numpy as np
import scipy as sp
import scipy.stats as stats
from metrics.munkres import Munkres
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from metrics.r2 import linear_disentanglement, nonlinear_disentanglement
from sklearn.preprocessing import StandardScaler


def correlation(x, y, method='Pearson'):
    """Evaluate correlation
     Args:
         x: data to be sorted
         y: target data
     Returns:
         corr_sort: correlation matrix between x and y (after sorting)
         sort_idx: sorting index
         x_sort: x after sorting
         method: correlation method ('Pearson' or 'Spearman')
     """

    x = x.copy()
    y = y.copy()
    dim = x.shape[0]

    if method=='Pearson':
        corr = np.corrcoef(y, x)
        corr = corr[0:dim,dim:]
    elif method=='Spearman':
        corr, pvalue = stats.spearmanr(y.T, x.T)
        corr = corr[0:dim, dim:]

    munk = Munkres()
    indexes = munk.compute(-np.absolute(corr))

    sort_idx = np.zeros(dim)
    x_sort = np.zeros(x.shape)
    for i in range(dim):
        sort_idx[i] = indexes[i][1]
        x_sort[i,:] = x[indexes[i][1],:]

    if method=='Pearson':
        corr_sort = np.corrcoef(y, x_sort)
        corr_sort = corr_sort[0:dim,dim:]
    elif method=='Spearman':
        corr_sort, pvalue = stats.spearmanr(y.T, x_sort.T)
        corr_sort = corr_sort[0:dim, dim:]

    return corr_sort, sort_idx, x_sort


def compute_mcc(mus_train, ys_train, correlation_fn):
  """Computes score based on both training and testing codes and factors."""
  score_dict = {}
  result = np.zeros(mus_train.shape)
  result[:ys_train.shape[0],:ys_train.shape[1]] = ys_train
  
  for i in range(len(mus_train) - len(ys_train)):
    result[ys_train.shape[0] + i, :] = np.random.normal(size=ys_train.shape[1])
  corr_sorted, sort_idx, mu_sorted = correlation(mus_train, result, method=correlation_fn)
  mcc = np.mean(np.abs(np.diag(corr_sorted)[:len(ys_train)]))
  return mcc


def correlation2(mus, ys, method="Pearson"):
    """
    Compute correlation matrix between mus and ys, return optimal matching using Munkres.
    mus: shape [D1, N]
    ys: shape [D2, N]
    """
    D1, N1 = mus.shape
    D2, N2 = ys.shape
    assert N1 == N2, "Sample size (N) must match."

    if method == "Pearson":
        corr = np.corrcoef(mus, ys)[:D1, D1:]
    else:
        raise NotImplementedError(f"Method {method} not supported.")

    dim = min(D1, D2)
    corr_sub = corr[:dim, :dim]

    munk = Munkres()
    indexes = munk.compute(-np.abs(corr_sub))
    optimal_values = [corr_sub[i, j] for i, j in indexes]

    return corr_sub, indexes, np.mean(optimal_values)

def compute_mcc2(mus_train, ys_train, correlation_fn="Pearson", verbose=False):
    """Computes MCC score and prints which variables are matched."""
    score_dict = {}
    result = np.zeros(mus_train.shape)
    result[:ys_train.shape[0], :ys_train.shape[1]] = ys_train
    for i in range(mus_train.shape[0] - ys_train.shape[0]):
        result[ys_train.shape[0] + i, :] = np.random.normal(size=ys_train.shape[1])
    corr_sorted, sort_idx, mu_sorted = correlation_safe(mus_train, result, method=correlation_fn)

    if verbose:
        print("Matching (meta i -> latent j):")
        for i, j in enumerate(sort_idx.astype(int)):
            print(f"y{i} -> z{j}, corr = {corr_sorted[i, i]:.4f}")

    diag_corrs = np.abs(np.diag(corr_sorted)[:ys_train.shape[0]])
    if verbose:
        print("Diagonal correlations used for MCC:", diag_corrs)
    mcc = np.mean(diag_corrs)
    return mcc

def compute_mcc_topk(mus_train, ys_train, k=None, correlation_fn='Pearson'):
    """
    Computes top-k MCC score using the k highest correlation coefficients.
    
    Args:
        mus_train: latent representations, shape [D1, N]
        ys_train: ground truth factors, shape [D2, N]
        k: number of top correlations to use for averaging. If None, uses min(D1, D2)
        correlation_fn: correlation method ('Pearson' or 'Spearman')
        
    Returns:
        topk_mcc: mean of top-k absolute correlation coefficients
        all_correlations: sorted absolute correlations (descending)
        matching_info: list of (y_idx, z_idx, correlation) for top-k matches
    """
    
    result = np.zeros(mus_train.shape)
    result[:ys_train.shape[0], :ys_train.shape[1]] = ys_train
    
    for i in range(mus_train.shape[0] - ys_train.shape[0]):
        result[ys_train.shape[0] + i, :] = np.random.normal(size=ys_train.shape[1])
    
    corr_sorted, sort_idx, mu_sorted = correlation_safe(mus_train, result, method=correlation_fn)
    
    diag_corrs = np.abs(np.diag(corr_sorted)[:ys_train.shape[0]])
    
    if k is None:
        k = min(mus_train.shape[0], ys_train.shape[0])
    else:
        k = min(k, len(diag_corrs))  
    
    sorted_indices = np.argsort(diag_corrs)[::-1]  
    topk_correlations = diag_corrs[sorted_indices[:k]]
    topk_mcc = np.mean(topk_correlations)
    
    matching_info = []
    for i in range(k):
        orig_idx = sorted_indices[i]
        matched_latent_idx = int(sort_idx[orig_idx])
        correlation_val = diag_corrs[orig_idx]
        matching_info.append((orig_idx, matched_latent_idx, correlation_val))
    
    return topk_mcc




def correlation_safe(x, y, method='Pearson'):
    """Evaluate correlation with robust handling of zero std.

    Args:
        x: data to be sorted, shape (dim, n_samples)
        y: target data, shape (dim, n_samples)
        method: 'Pearson' or 'Spearman'

    Returns:
        corr_sort: sorted correlation matrix
        sort_idx: index mapping from y_i to x_j
        x_sort: x reordered according to best match
    """
    print(x.shape, y.shape)
    x = x.copy()
    y = y.copy()
    dim = x.shape[0]
    eps = 1e-15  

    def safe_pearson(a, b):
        a_mean = a.mean(axis=1, keepdims=True)
        b_mean = b.mean(axis=1, keepdims=True)
        a_std = a.std(axis=1, keepdims=True)
        b_std = b.std(axis=1, keepdims=True)

        a_std[a_std < eps] = eps
        b_std[b_std < eps] = eps

        a_norm = (a - a_mean) / a_std
        b_norm = (b - b_mean) / b_std

        return (a_norm @ b_norm.T) / a.shape[1]

    if method == 'Pearson':
        corr = safe_pearson(y, x)
    elif method == 'Spearman':
        corr, _ = stats.spearmanr(y.T, x.T)
        corr = corr[:dim, dim:]

   
    munk = Munkres()
    indexes = munk.compute(-np.abs(corr)) 
    
    sort_idx = np.zeros(dim, dtype=int)
    x_sort = np.zeros_like(x)
    for i in range(dim):
        sort_idx[i] = indexes[i][1]
        x_sort[i, :] = x[sort_idx[i], :]

    corr_sort = np.zeros_like(corr)
    if method == 'Pearson':
        corr_sort = safe_pearson(y, x_sort)
    elif method == 'Spearman':
        corr_sort, _ = stats.spearmanr(y.T, x_sort.T)
        corr_sort = corr_sort[:dim, dim:]

    return corr_sort, sort_idx, x_sort
#%% MMD
'''
Maximum Mean Discrepancy: https://blog.csdn.net/sinat_34173979/article/details/105876584
'''
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0) 
    
    total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2) 

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    return sum(kernel_val) 


def plot_mcc_matrix(corr_matrix, title="MCC Matrix", xlabel="Latent Variables", ylabel="Metadata Variables", figsize=(8, 6)):
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0,
                xticklabels=[f"z{i}" for i in range(corr_matrix.shape[1])],
                yticklabels=[f"y{i}" for i in range(corr_matrix.shape[0])])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=300)
    plt.show()


def compute_mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul,
                             	kernel_num=kernel_num,
                              fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size] # Source<->Source
    YY = kernels[batch_size:, batch_size:] # Target<->Target
    XY = kernels[:batch_size, batch_size:] # Source<->Target
    YX = kernels[batch_size:, :batch_size] # Target<->Source
    loss = torch.mean(XX + YY - XY -YX) 
    return loss


