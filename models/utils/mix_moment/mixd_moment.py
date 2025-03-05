import numpy as np
from scipy.stats import mstats_basic


def mixed_moment(x, y, orderx, ordery, axis=None, nan_policy='propagate'):
    """
    Calculate the mixed moments of two random variables.

    Parameters:
    - x, y: array_like
        Input arrays representing two random variables.
    - order: int, optional
        Order of the mixed moment. Default is 2.
    - axis: int or None, optional
        Axis along which the mixed moment is computed. Default is None.
    - nan_policy: {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains NaN.
        The following options are available (default is 'propagate'):
          * 'propagate': returns nan
          * 'raise': throws an error
          * 'omit': performs the calculations ignoring nan values

    Returns:
    - mixed_moment: ndarray or float
        The mixed moment along the given axis or over all values if axis
        is None.
    """
    x, axis_x = np.asarray(x), axis
    y, axis_y = np.asarray(y), axis

    contains_nan, nan_policy = np.any(np.isnan(x)), nan_policy
    contains_nan_y, nan_policy_y = np.any(np.isnan(y)), nan_policy

    if contains_nan and nan_policy == 'omit':
        x = np.ma.masked_invalid(x)
        x = mstats_basic.moment(x, orderx, axis_x)

    if contains_nan_y and nan_policy_y == 'omit':
        y = np.ma.masked_invalid(y)
        y = mstats_basic.moment(y, ordery, axis_y)

    mixed = np.mean(x**orderx * y**ordery, axis=axis_x)

    return mixed


def self_moments(x, orders, axis=None, nan_policy='propagate'):
    """
    Calculate moments about the mean for a sample at specified orders.

    Parameters:
    - x: array_like
        Input array representing a random variable.
    - orders: array_like of ints
        Orders of the moments to be calculated.
    - axis: int or None, optional
        Axis along which the moments are computed. Default is None.
    - nan_policy: {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains NaN.
        The following options are available (default is 'propagate'):
          * 'propagate': returns nan
          * 'raise': throws an error
          * 'omit': performs the calculations ignoring nan values

    Returns:
    - moments: ndarray
        Array of moments corresponding to the specified orders.
    """
    x, axis_x = np.asarray(x), axis

    contains_nan, nan_policy = np.any(np.isnan(x)), nan_policy

    if contains_nan and nan_policy == 'omit':
        x = np.ma.masked_invalid(x)
        moments = [mstats_basic.moment(x, order, axis_x) for order in orders]
    else:
        moments = [np.mean(x**order, axis=axis_x) for order in orders]

    return np.array(moments)
