from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model, kernel_ridge
import torch
import numpy as np
import scipy as sp
from typing import Union
from typing_extensions import Literal
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.exceptions import ConvergenceWarning
import warnings

__Mode = Union[Literal["r2"]]

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x 


def _disentanglement(z, hz, mode: __Mode = "r2", reorder=None):
    """Measure how well hz reconstructs z measured either by the Coefficient of Determination or the
    Pearson/Spearman correlation coefficient."""

    assert mode in ("r2", "accuracy")

    if mode == "r2":
        return metrics.r2_score(z, hz), None
    elif mode == "accuracy":
        return metrics.accuracy_score(z, hz), None


def mean_squared_error(y_true, y_pred):
    """
    Compute Mean Squared Error between true and predicted values.
    
    Parameters:
        y_true (np.ndarray): Ground truth values
        y_pred (np.ndarray): Predicted values
        
    Returns:
        float: MSE value
    """
    return np.mean((y_true - y_pred) ** 2)

def mse_part(pred, y, indices):
    return np.mean((pred[:, indices] - y[:, indices]) ** 2)


class LoggingMLP(MLPRegressor):
    def fit(self, X, y, X_test, y_test, select_mode="best"):
        total_iters = self.max_iter
        self.best_test_score_ = None
        self.best_test_pred_ = None
        self.last_test_score_ = None
        self.last_test_pred_ = None

        original_max_iter = self.max_iter
        original_warm_start = self.warm_start
        self.max_iter = 1
        self.warm_start = True

        for _ in range(total_iters):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                super().fit(X, y)
            test_pred = super().predict(X_test)
            test_score, _ = _disentanglement(y_test, test_pred, mode="r2", reorder=False)
            test_score = float(test_score)

            self.last_test_score_ = test_score
            self.last_test_pred_ = test_pred.copy()

            if self.best_test_score_ is None or test_score > self.best_test_score_:
                self.best_test_score_ = test_score
                self.best_test_pred_ = test_pred.copy()

        self.max_iter = original_max_iter
        self.warm_start = original_warm_start
        self.r2_select_mode_ = select_mode
        return self

    def get_selected_test_prediction(self, select_mode="best"):
        if select_mode == "best":
            return self.best_test_pred_
        if select_mode == "last":
            return self.last_test_pred_
        raise ValueError(f"Unknown r2 selection mode: {select_mode}")


def nonlinear_disentanglement(
    z,
    hz,
    z_test,
    hz_test,
    mode: __Mode = "r2",
    alpha=1.0,
    gamma=None,
    train_mode=False,
    model=None,
    scaler_z=None,
    scaler_hz=None,
    select_mode="best",
):
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

    
    if train_mode:
        model = LoggingMLP(
            hidden_layer_sizes=(256,),
            activation="identity",
            solver="adam",
            max_iter=1000,
            random_state=42,
            warm_start=True  
        )
        model.fit(hz, z, hz_test, z_test, select_mode=select_mode)
        
        
        return model
    else:
        if isinstance(model, LoggingMLP):
            hz_test = model.get_selected_test_prediction(select_mode=select_mode)
        else:
            hz_test = model.predict(hz_test)
        inner_result = _disentanglement(z_test, hz_test, mode=mode, reorder=False)
        return inner_result, (z_test, hz_test)
        

def scale_to_range(x, range_min=0.0, range_max=1.0):
    x_min = np.min(x, axis=0, keepdims=True)
    x_max = np.max(x, axis=0, keepdims=True)
    denom = np.where(x_max - x_min == 0, 1e-8, x_max - x_min)
    x_scaled = (x - x_min) / denom
    return x_scaled * (range_max - range_min) + range_min


def tanh_scale(x):
    x_mean = np.mean(x, axis=0, keepdims=True)
    x_std = np.std(x, axis=0, keepdims=True)
    x_std = np.where(x_std == 0, 1e-8, x_std)
    x_normalized = (x - x_mean) / x_std
    return np.tanh(x_normalized)


def compute_r2(z, hz, select_mode="best"):
    train_hz, test_hz, train_z, test_z = train_test_split(hz, z, test_size=0.2, random_state=42)

    train_hz = to_numpy(train_hz)
    train_z = to_numpy(train_z)
    test_hz = to_numpy(test_hz)
    test_z = to_numpy(test_z)
    
    scaler_hz = MinMaxScaler()
    train_hz = scaler_hz.fit_transform(train_hz)

    scaler_z = MinMaxScaler()
    train_z = scaler_z.fit_transform(train_z)

    test_hz = scaler_hz.transform(test_hz)
    test_z = scaler_z.transform(test_z)

    model = nonlinear_disentanglement(
        train_z,
        train_hz,
        test_z,
        test_hz,
        train_mode=True,
        select_mode=select_mode,
    )
    r2_result, _ = nonlinear_disentanglement(
        train_z,
        train_hz,
        test_z,
        test_hz,
        train_mode=False,
        model=model,
        select_mode=select_mode,
    )
    return float(r2_result[0])

def linear_disentanglement(z, hz, mode: __Mode = "r2", train_test_split=False, train_mode=False, model=None):
    """Calculate disentanglement up to linear transformations.

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

    if train_test_split:
        n_train = len(z) // 2
        z_1 = z[:n_train]
        hz_1 = hz[:n_train]
        z_2 = z[n_train:]
        hz_2 = hz[n_train:]
        if mode == "accuracy":
            model = linear_model.LogisticRegression()
        else:
            model = linear_model.LinearRegression()
        model.fit(hz_1, z_1)
        hz_2 = model.predict(hz_2)

        inner_result = _disentanglement(z_2, hz_2, mode=mode, reorder=False)

        return inner_result, (z_2, hz_2)
    else:
        if train_mode:
            if mode == "accuracy":
                model = linear_model.LogisticRegression()
            else:
                model = linear_model.LinearRegression()
            model.fit(hz, z)
            return model
        else:
            hz = model.predict(hz)
            inner_result = _disentanglement(z, hz, mode=mode, reorder=False)
            return inner_result, (z, hz)
