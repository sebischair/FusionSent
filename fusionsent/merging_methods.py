# This module provides different functionalities for merging two sets of model parameters.

from typing import Union
import numpy as np
import torch

def merge_models(
        model_state_dict0,
        model_state_dict1,
        merging_method: str='slerp',
        t: Union[float, np.ndarray]=0.5,
        DOT_THRESHOLD: float = 0.9995, 
        eps: float = 1e-8
    ):
    """
    Merges two model state dictionaries using a specified merging method.

    Args:
        model_state_dict0: State dictionary of the first model.
        model_state_dict1: State dictionary of the second model.
        merging_method (str): Method to be used for merging (either 'slerp' [default], or 'lerp').
        t (Union[float, np.ndarray]): Interpolation factor, can be a float or ndarray.
        DOT_THRESHOLD (float): Threshold to consider vectors as collinear (used only if merging_method = 'slerp').
        eps (float): Small value to prevent division by zero (used only if merging_method = 'slerp').

    Returns:
        fused_parameter_dict (dict): Dictionary containing the merged parameters.
    """
    fused_parameter_dict = {}
    if merging_method  == 'slerp':
        for key in model_state_dict1:
            fused_parameter_dict[key] = _slerp(t=t, v0=model_state_dict0[key], v1=model_state_dict1[key], DOT_THRESHOLD=DOT_THRESHOLD, eps=eps)        
    elif merging_method == 'lerp':
        for key in model_state_dict1:
            fused_parameter_dict[key] = _lerp(t=t, v0=model_state_dict0[key], v1=model_state_dict1[key])
    else:
        raise ValueError(f"'merging_method' has unsupported value '{merging_method}'. Choose either 'slerp' or 'lerp'.")     

    return fused_parameter_dict

def _lerp(
    t: float,
    v0: Union[np.ndarray, torch.Tensor],
    v1: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Traditional linear interpolation of model parameters as simple weighted average.

    From: https://github.com/cg123/mergekit#linear
    Args:
        t (float/np.ndarray): Float value between 0.0 and 1.0 as interpolation or weighting factor. At t=0 will return v0, at t=1 will return v1.
        v0 (np.ndarray): Starting vector
        v1 (np.ndarray): Final vector
        DOT_THRESHOLD (float): Threshold for considering the two vectors as colinear. Not recommended to alter this.
    Returns:
        v2 (np.ndarray or torch.Tensor, depending on the input vectors): Interpolation vector between v0 and v1
    """
    return (1 - t) * v0 + t * v1

def _slerp(
    t: Union[float, np.ndarray],
    v0: Union[np.ndarray, torch.Tensor],
    v1: Union[np.ndarray, torch.Tensor],
    DOT_THRESHOLD: float = 0.9995,
    eps: float = 1e-8,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Spherical Linear Interpolation (SLERP) is a method used to smoothly interpolate between two vectors (i.e. model parameters). It maintains a constant rate of change and preserves the geometric properties of the spherical space in which the vectors reside.

    SLERP is implemented using the following steps:

    1. Normalize the input vectors to unit length, ensuring they represent directions rather than magnitudes
    2. Calculate the angle between these vectors using their dot product.
    3. If the vectors are nearly collinear, it defaults to linear interpolation for efficiency. Otherwise, SLERP computing scale factors based on the interpolation factor t (t=0 = 100% of the first vector, t=1 = 100% of model 2) and the angle between the vectors.
    4. These factors are used to weigh the original vectors, which are then summed to obtain the interpolated vector.

    There are several reasons to prefer SLERP over a traditional linear interpolation. For example, in high-dimensional spaces, linear interpolation can lead to a decrease in the magnitude of the interpolated vector (i.e., it reduces the scale of weights). Moreover, the change in direction of the weights often represents more meaningful information (like feature learning and representation) than the magnitude of change.

    From: https://github.com/cg123/mergekit#slerp
    Args:
        t (float/np.ndarray): Float value between 0.0 and 1.0 as interpolation or weighting factor. At t=0 will return v0, at t=1 will return v1.
        v0 (np.ndarray): Starting vector
        v1 (np.ndarray): Final vector
        DOT_THRESHOLD (float): Threshold for considering the two vectors as colinear. Not recommended to alter this.
    Returns:
        v2 (np.ndarray or torch.Tensor, depending on the input vectors): Interpolation vector between v0 and v1
    """
    is_torch = False
    if not isinstance(v0, np.ndarray):
        is_torch = True
        v0 = v0.detach().cpu().float().numpy()
    if not isinstance(v1, np.ndarray):
        is_torch = True
        v1 = v1.detach().cpu().float().numpy()

    # Copy the vectors to reuse them later
    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)

    # Normalize the vectors to get the directions and angles
    v0 = _normalize(v0, eps)
    v1 = _normalize(v1, eps)

    # Dot product with the normalized vectors (can't use np.dot in W)
    dot = float(np.sum(v0 * v1))

    # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
    if np.abs(dot) > DOT_THRESHOLD:
        res = _lerp(t, v0_copy, v1_copy)
        return _maybe_torch(res, is_torch)

    # Calculate initial angle between v0 and v1
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)

    # Finish the slerp algorithm
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    res = s0 * v0_copy + s1 * v1_copy

    return _maybe_torch(res, is_torch)

def _maybe_torch(v: np.ndarray, is_torch: bool) -> Union[np.ndarray, torch.Tensor]:
    if not isinstance(v, np.ndarray):
        v = np.array(v)
    if is_torch:
        return torch.from_numpy(v)
    return v

def _normalize(v: np.ndarray, eps: float) -> np.ndarray:
    norm_v = np.linalg.norm(v)
    if norm_v > eps:
        v = v / norm_v
    return v