from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn import kernel_ridge
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import scipy as sp
from typing import Union
from typing_extensions import Literal
import ipdb as pdb

__Mode = Union[Literal["r2"]]


def _disentanglement(z, hz, mode: __Mode = "r2", reorder=None):
    """Measure how well hz reconstructs z measured either by the Coefficient of Determination or the
    Pearson/Spearman correlation coefficient."""

    assert mode in ("r2", "accuracy")

    if mode == "r2":
        return metrics.r2_score(z, hz), None
    elif mode == "accuracy":
        return metrics.accuracy_score(z, hz), None

def nonlinear_disentanglement(z, hz, mode: __Mode = "r2", alpha=1.0, gamma=None, train_mode=False, model=None, scaler_z=None, scaler_hz=None):
    """Calculate disentanglement up to nonlinear transformations.

    Args:
        z: Ground-truth latents.
        hz: Reconstructed latents.
        mode: Can be r2, pearson, spearman
        train_test_split: Use first half to train linear model, second half to test.
            Is only relevant if there are less samples then latent dimensions.
    """
    if torch.is_tensor(hz):
        hz = hz.detach().cpu().numpy()
    if torch.is_tensor(z):
        z = z.detach().cpu().numpy()

    assert isinstance(z, np.ndarray), "Either pass a torch tensor or numpy array as z"
    assert isinstance(hz, np.ndarray), "Either pass a torch tensor or numpy array as hz"

    # # split z, hz to get train and test set for linear model
    # if train_test_split:
    #     n_train = len(z) // 2
    #     z_1 = z[:n_train]
    #     hz_1 = hz[:n_train]
    #     z_2 = z[n_train:]
    #     hz_2 = hz[n_train:]
    #     model = kernel_ridge.KernelRidge(kernel='linear', alpha=alpha, gamma=gamma)
    #     model.fit(hz_1, z_1)
    #     hz_2 = model.predict(hz_2)

    #     inner_result = _disentanglement(z_2, hz_2, mode=mode, reorder=False)

    #     return inner_result, (z_2, hz_2)
    # else:
    if train_mode:
        model = GridSearchCV(
            kernel_ridge.KernelRidge(kernel='rbf', gamma=0.1),
            param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 4)}, 
            cv=3, 
            n_jobs=-1
        )
        model.fit(hz, z)
        return model
    else:
        hz = model.predict(hz)
        inner_result = _disentanglement(z, hz, mode=mode, reorder=False)
        return inner_result, (z, hz)


def compute_r2(z, hz):
    # split the two 
    n_train = len(z) // 2
    train_z = z[:n_train]
    train_hz = hz[:n_train]
    test_z = z[n_train:]
    test_hz = hz[n_train:]

    # normalize the first half and keep the normalization constants
    scaler_hz = StandardScaler()
    train_hz = scaler_hz.fit_transform(train_hz.detach().cpu().numpy())
    scaler_z = StandardScaler()
    train_z = scaler_z.fit_transform(train_z.detach().cpu().numpy())

    # normalize the second one with the normalization constants
    test_hz = scaler_hz.transform(test_hz.detach().cpu().numpy())
    test_z = scaler_z.transform(test_z.detach().cpu().numpy())

    # train the model with the normalized first half
    model = nonlinear_disentanglement(train_z, train_hz, train_mode=True)

    # apply the model to infer on the second normalized half.
    r2, _ = nonlinear_disentanglement(test_z, test_hz, model=model, train_mode=False)

    return r2[0]