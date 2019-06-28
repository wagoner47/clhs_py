#!/usr/bin/env python
"""Conditioned Latin Hypercube Sampling

This is a python implementation of the conditions LHS of Minasny & McBratney
(2006; C&G 32 1378).

I've also drawn heavily from the clhs R package of Roudier, Brugnard, Beaudette,
and Louis (CRAN entry: https://CRAN.R-project.org/package=clhs; github:
https://github.com/cran/clhs), but did not implement the cost option of their
code, nor the DLHS option of Minasny & McBratney (2010)
"""
import numpy as np
import copy
import warnings

__all__ = ["get_strata", "get_correlation_matrix", "get_random_samples",
           "counts_matrix", "continuous_objective_func",
           "correlation_objective_func", "clhs_objective_func",
           "resample_random", "resample_worst", "clhs"]
__author__ = "Erika Wagoner"
__copyright__ = "Copyright 2019, Erika Wagoner"
__credits__ = ["Erika Wagoner"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Erika Wagoner"
__email__ = "wagoner47+clhs@email.arizona.edu"
__status__ = "Development"


def get_strata(predictors, num_samples, good_mask=None):
    """Get sampling strata

    Get the quantiles of ``predictors`` as the sampling strata. This assumes
    that ``num_samples`` is already less than the number of (possibly masked)
    rows (the size along the 0th axis): this condition is not checked! The
    return is actually the ``num_samples + 1`` edges of the strata

    Parameters
    ----------
    predictors : array-like (Nrows,) or (Nrows, Npredictors)
        The input data to stratify, which may be 1D or 2D
    num_samples : `int`
        The number of samples being drawn, which is the number of strata that
        will be created. Must be at least 1, but should be less than the
        number of rows in ``predictors``
    good_mask : array-like[`bool`] (Nrows,), optional
        A mask for selecting data, where `True` indicates good data and `False`
        indicates data to be excluded. If `None` (default), all data is assumed
        to be good

    Returns
    -------
    strata : :class:`numpy.ndarray`[`float`]
        The edges of the strata. If ``predictors`` is 1D, the result is also 1D
        with size ``num_samples + 1``. Otherwise, the shape is
        (``num_samples + 1``, Npredictors)

    Raises
    ------
    ValueError
        Raised if ``num_samples`` is less than 1
    """

    num_samples = int(num_samples)
    if good_mask is None:
        good_mask = np.ones(np.shape(np.squeeze(predictors))[0], dtype=bool)
    if num_samples < 1:
        raise ValueError("Invalid number of samples: {}".format(num_samples))
    if np.ndim(np.squeeze(predictors)) == 1:
        return np.quantile(np.squeeze(predictors)[good_mask],
                           np.linspace(0, 1, num=(num_samples + 1)))
    else:
        return np.quantile(np.squeeze(predictors)[good_mask],
                           np.linspace(0, 1, num=(num_samples + 1)),
                           axis=0)


def get_correlation_matrix(predictors, good_mask=None):
    """Get the correlation matrix of the predictors

    This can handle 1D or 2D predictors

    Parameters
    ----------
    predictors : array-like (Nrows,) or (Nrows, Npredictors)
        The data for which to find the correlation matrix. 2D inputs should
        have columns (1st axis) corresponding to different predictors
    good_mask : array-like[`bool`] (Nrows,), optional
        A mask for selecting data, where `True` indicates good data and `False`
        indicates data to be excluded. If `None` (default), all data is assumed
        to be good

    Returns
    -------
    corr : `float` or :class:`numpy.ndarray`[`float`] (Npredictors, Npredictors)
        The correlation matrix, which is a scalar when ``predictors`` is 1D or
        when it's length along the 1st axis is 1
    """
    if good_mask is None:
        good_mask = np.ones(np.shape(np.squeeze(predictors))[0], dtype=bool)
    if np.ndim(np.squeeze(predictors)) == 1:
        return np.corrcoef(np.squeeze(predictors)[good_mask])
    else:
        return np.corrcoef(np.squeeze(predictors)[good_mask], rowvar=False)


def get_random_samples(num_data, num_samples, good_mask=None, include=None):
    """Generate random sample of indices

    Randomly choose indices as the initialization for the cLHS. The optional
    parameter ``include`` gives an index or indices which must be in the
    selected indices

    Parameters
    ----------
    num_data : `int`
        The number of rows (or observations) in the data
    num_samples : `int`
        The number of samples to draw, which must be less than ``num_data``
    good_mask : array-like[`bool`] (Nrows,), optional
        A mask for selecting data, where `True` indicates good data and `False`
        indicates indices to be excluded. If `None` (default), all indices are
        assumed to be valid
    include : `int` or array-like[`int`], optional
        Indices that must be included in the sample, if any

    Returns
    -------
    sampled_indices : :class:`numpy.ndarray`[`int`] (``num_samples``,)
        The indices included in the random sampling
    remaining_indices : :class:`numpy.ndarray`[`int`]
        The indices not in the sampling, which will never contain any indices
        from ``include`` (if given). This will be 1D with length
        ``num_data - num_samples`` if all indices are valid, or
        ``num_data - num_samples - np.count_nonzero(~good_mask)``
        if any indices are masked out
    """
    if good_mask is None:
        good_mask = np.ones(num_data, dtype=bool)
    masked_indices = np.arange(num_data, dtype=int)[good_mask]
    if include is None:
        include = []
    always_include_ = np.unique(include).astype(int)
    num_always_include = always_include_.size
    select_size = num_samples - num_always_include
    available_indices = masked_indices[np.isin(
        masked_indices, always_include_, invert=True, assume_unique=True)]
    sampled_indices = np.append(
        np.random.choice(available_indices, select_size, replace=False),
        always_include_)
    remaining_indices = masked_indices[np.isin(
        masked_indices, sampled_indices, invert=True, assume_unique=True)]
    return sampled_indices, remaining_indices


def counts_matrix(x, quantiles):
    """Count samples in strata

    Get eta, the number of samples in ``x`` binned by ``quantiles`` in each
    variable, for continuous variables. The shape of eta is the same as the
    shape of ``x``, and the shape of ``quantiles`` should be
    (``numpy.shape(x)[0] + 1``, ``numpy.shape(x)[1]``) for 2D, or
    (``numpy.size(x) + 1``,) for 1D

    Parameters
    ----------
    x : :class:`numpy.ndarray` (Nx,) or (Nx, Npredictors)
        The sampled predictors, with observations as rows and predictors (if
        more than 1) as columns
    quantiles : :class:`numpy.ndarray` (Nx + 1,) or (Nx + 1, Npredictors)
        The quantiles which mark the edge of strata. The 0th axis must be
        one element longer than the 0th axis of ``x``

    Returns
    -------
    eta : :class:`numpy.ndarray`[`int`] (Nx,) or (Nx, Npredictors)
        The matrix of counts in strata, with the same shape as ``x``
    """
    x_ = np.squeeze(x)
    q_ = np.squeeze(quantiles)
    if x_.ndim == 1:
        eta = np.histogram(x_, bins=q_)[0].astype(int)
    else:
        eta = np.array([
            np.histogram(xj, bins=qj)[0].astype(int) for xj, qj in zip(
                x_.T, q_.T)]).T
    return eta


def continuous_objective_func(x, quantiles):
    """Calculate the objective function for the counts in strata

    This finds the objective function for continuous (not categorical)
    predictors in each of the stratum by summing the contribution of all
    predictors in the stratum. The actual objective function for the
    continuous predictors is found by adding all elements of the return of
    this function.

    Parameters
    ----------
    x : :class:`numpy.ndarray` (Nx,) or (Nx, Npredictors)
        The sampled predictors, with observations as rows and predictors (if
        more than 1) as columns
    quantiles : :class:`numpy.ndarray` (Nx + 1,) or (Nx + 1, Npredictors)
        The quantiles which mark the edge of strata. The 0th axis must be
        one element longer than the 0th axis of ``x``

    Returns
    -------
    o_in_strata : :class:`numpy.ndarray`[`int`] (Nx,)
        The objective function value in each of the strata

    Notes
    -----
    The objective function for the continuous variables is defined in Minasny &
    McBratney (2006) [1]_ for :math:`n` samples and :math:`k` predictors as
    .. math::

        O_1 = \sum_i^n \sum_{j = 1}^k \left\lvert \eta\left(q_j^i \leq x_j <
        q_j^{i+1}\right\rvert\: ,
    where :math:`\eta\left(q_j^i \leq x_j < q_j^{i+1}\right)` is the number of
    samples :math:`x_j` of a given predictor that fall within the quantile bin
    :math:`q_j^i` and :math:`q_j^{i+1}`.

    References
    ----------
    .. [1] B. Minasny, A. B. McBratney, "A conditioned Latin hypercube method
    for sampling in the presence of ancillary information", Computers &
    Geosciences, vol. 32, pp. 1378-1388, 2006.
    """
    eta = counts_matrix(x, quantiles)
    return np.sum(np.abs(eta - 1), axis=1)


def correlation_objective_func(data_corr, x):
    """Objective function contribution of the correlation matrix

    Calculate the objective function for the correlation matrix of the
    continuous variables

    Parameters
    ----------
    data_corr : `float` or :class:`numpy.ndarray`[`float`] (Nx, Nx)
        The correlation matrix (or scalar) of the full (not sampled)
        continuous predictors
    x : :class:`numpy.ndarray`[`float`] (Nx,) or (Nx, Npredictors)
        The sampled continuous predictors

    Returns
    -------
    o_corr : `float`
        The contribution to the objective function from the correlation of the
        predictors

    Notes
    -----
    The objective function for the correlation matrix is defined in Minasny &
    McBratney (2006) [1]_ for :math:`k` predictors as
    .. math::

        O_3 = \sum_{i = 1}^k \sum_{j = 1}^k \left\lvert C_{ij} - T_{ij}
        \right\rvert\: ,
    where :math:`C_{ij}` is the element in the :math:`i`'th row and :math:`j`'th
    column of the correlation matrix of the full predictors, and :math:`T_{ij}`
    is the same for the correlation matrix of the sampled predictors.

    References
    ----------
    .. [1] B. Minasny, A. B. McBratney, "A conditioned Latin hypercube method
    for sampling in the presence of ancillary information", Computers &
    Geosciences, vol. 32, pp. 1378-1388, 2006.
    """
    x_corr = get_correlation_matrix(x)
    return np.sum(np.abs(data_corr - x_corr))


def clhs_objective_func(x, quantiles, data_corr, weights=None):
    """Full objective function

    Get the value of the full objective function. Note that at this time,
    including categorical data has not yet been implemented, but should be
    included in the future (for, e.g., star-galaxy separation).

    Parameters
    ----------
    x : :class:`numpy.ndarray`[`float`] (Nx,) or (Nx, Npredictors)
        The sampled continuous predictors
    quantiles : :class:`numpy.ndarray` (Nx + 1,) or (Nx + 1, Npredictors)
        The quantiles which mark the edge of strata. The 0th axis must be
        one element longer than the 0th axis of ``x``
    data_corr : `float` or :class:`numpy.ndarray`[`float`] (Nx, Nx)
        The correlation matrix (or scalar) of the full (not sampled)
        continuous predictors
    weights : array-like[`float`] (3,), optional
        The weights for each of the objective function contributions, ordered
        (1) continuous, (2) categorical, (3) correlation. If `None` (default),
        applies a weight of 1 to all contributions, which is fine for general
        applications

    Returns
    -------
    obj_tot : `float`
        The total objective function for the cLHS
    obj_continuous : :class:`numpy.ndarray`[`float`] (Nx,)
        The continuous variable objective function in strata,
        see :func:`~.continuous_objective_func`
    obj_categorical : `float`
        The categorical objective function, currently 0 as this is not yet
        implemented
    obj_corr : `float`
        The correlation matrix objective function,
        see :func:`~.correlation_objective_func`

    Notes
    -----
    The weight for the categorical objective function must be given if
    ``weights`` is given, even though this is not yet implemented.

    The total objective function is defined in Minasny &
    McBratney (2006) [1]_ as
    .. math::

        O = w_1 O_1 + w_2 O_2 + w_3 O_3\: ,
    where :math:`O_1` is the objective function for continuous predictors,
    :math:`O_2` is the objective function for categorical predictors,
    :math:`O_3` is the objective function for the correlation matrix, and the
    :math:`w_i` are the corresponding weights.

    References
    ----------
    .. [1] B. Minasny, A. B. McBratney, "A conditioned Latin hypercube method
    for sampling in the presence of ancillary information", Computers &
    Geosciences, vol. 32, pp. 1378-1388, 2006.
    """
    if weights is not None:
        weights_ = np.asarray(weights)
    else:
        weights_ = np.ones(3)
    obj_continuous = continuous_objective_func(x, quantiles)
    obj_categorical = 0
    obj_corr = correlation_objective_func(data_corr, x)
    obj_all = np.array([
        np.sum(obj_continuous), np.sum(obj_categorical), obj_corr])
    return np.sum(weights_ * obj_all), obj_continuous, obj_categorical, obj_corr


def resample_random(sampled_indices, remaining_indices, include=None):
    """Do random replacement of one item in the sample, but don't remove any
    indices in ``include``

    Parameters
    ----------
    sampled_indices : :class:`numpy.ndarray`[`int`] (Nsamp,)
        The indices of the previously selected sampling
    remaining_indices : :class:`numpy.ndarray`[`int`] (Nrem,)
        The indices which were not sampled (and which were not masked out),
        which should not have any indices from ``include`` (if given)
    include : `int` or array-like[`int`]
        Indices that must be in the sample, if any

    Returns
    -------
    new_sampled_indices : :class:`numpy.ndarray`[`int`] (Nsamp,)
        The indices of the new sampling
    new_remaining_indices : :class:`numpy.ndarray`[`int`] (Nrem,)
        The remaining indices not in the new sample (and not masked out). These
        won't contain anything from ``include`` (if given) as long as
        ``remaining_indices`` did not contain anything from ``include``
    """
    new_remaining_indices = remaining_indices.copy()
    if include is None:
        include = []
    always_include_ = np.unique(include)
    allowed_to_remove_indices = sampled_indices[
        np.isin(sampled_indices, always_include_, assume_unique=True,
                invert=True)].copy()
    idx_removed = np.random.choice(allowed_to_remove_indices.size)
    idx_added = np.random.choice(new_remaining_indices.size)
    sampled_index_to_remove = allowed_to_remove_indices[idx_removed]
    remained_index_to_sample = new_remaining_indices[idx_added]
    allowed_to_remove_indices[idx_removed] = remained_index_to_sample
    new_remaining_indices[idx_added] = sampled_index_to_remove
    new_sampled_indices = np.append(allowed_to_remove_indices, always_include_)
    return new_sampled_indices, new_remaining_indices


def resample_worst(continuous_objective_values, sampled_indices,
                   remaining_indices, include=None):
    """Resample in the worst stratum

    Remove a sampled item from the worst stratum (or one of the worst strata)
    and replace it with a random item from the unsampled items. But don't
    replace anything in ``include``

    Parameters
    ----------
    continuous_objective_values : array-like (Nsamp,)
        The continuous objective function in strata, as output by
        :func:`~.continuous_objective_func`
    sampled_indices : :class:`numpy.ndarray`[`int`] (Nsamp,)
        The indices of the previously selected sampling
    remaining_indices : :class:`numpy.ndarray`[`int`] (Nrem,)
        The indices which were not sampled (and which were not masked out),
        which should not have any indices from ``include`` (if given)
    include : `int` or array-like[`int`]
        Indices that must be in the sample, if any

    Returns
    -------
    new_sampled_indices : :class:`numpy.ndarray`[`int`] (Nsamp,)
        The indices of the new sampling
    new_remaining_indices : :class:`numpy.ndarray`[`int`] (Nrem,)
        The remaining indices not in the new sample (and not masked out). These
        won't contain anything from ``include`` (if given) as long as
        ``remaining_indices`` did not contain anything from ``include``
    """
    new_remaining_indices = remaining_indices.copy()
    if include is None:
        include = []
    always_include_ = np.unique(include)
    allowed_to_remove_indices = sampled_indices[
        np.isin(sampled_indices, always_include_, assume_unique=True,
                invert=True)].copy()
    unique_obj_vals, inverse_indices = np.unique(
        continuous_objective_values[allowed_to_remove_indices],
        return_inverse=True)
    idx_worst = np.arange(allowed_to_remove_indices.size)[inverse_indices[
        (inverse_indices == unique_obj_vals.size - 1)]]
    if np.size(idx_worst) > 1:
        idx_worst = np.random.choice(idx_worst)
    worst_sampled_index = allowed_to_remove_indices[idx_worst]
    idx_added = np.random.choice(new_remaining_indices.size)
    unsampled_index = new_remaining_indices[idx_added]
    allowed_to_remove_indices[idx_worst] = unsampled_index
    new_remaining_indices[idx_added] = worst_sampled_index
    new_sampled_indices = np.append(allowed_to_remove_indices, always_include_)
    return new_sampled_indices, new_remaining_indices


def clhs(predictors, num_samples, good_mask=None, include=None,
         max_iterations=10000, objective_func_limit=None,
         initial_temp=1, temp_decrease=0.95, cycle_length=10, p=0.5,
         weights=None, progress=True, random_state=None):
    """Generate conditioned Latin Hypercube sampling

    The full conditioned Latin Hypercube sampler. A progress bar may optionally
    be displayed, but may not be informative for large maximum number of
    iterations when a stopping criterion is given.

    Parameters
    ----------
    predictors : array-like (Nx,) or (Nx, Npredictors)
        The full set of predictors (independent data) with rows corresponding
        to observations and columns (if any) to individual predictors
    num_samples : `int`
        The number of samples to draw. Must be less than the number of rows
        in ``predictors``
    good_mask : array-like[`bool`] (Nx,), optional
        A mask for selecting data, where `True` indicates good data and `False`
        indicates data to be excluded. If `None` (default), all data is assumed
        to be good
    include : `int` or array-like[`int`], optional
        Indices that must be included in the sample, if any
    max_iterations : `int`, optional
        The maximum number of iterations, with a warning issued if
        convergence is not reached (based on the stopping criterion) within
        this number of iterations. Default 10,000
    objective_func_limit : `float`, optional
        The stopping criterion, if any. The code stops if the total objective
        function falls to or below this value. If `None` (default),
        the sample is allowed to run until ``max_iterations``
    initial_temp : `float`, optional
        The starting temperature for simulated annealing. This must be in the
        range (0, 1]. Default 1
    temp_decrease : `float`, optional
        The cooling factor, so that the temperature after cooling is this
        number times the temperature before cooling. Must be in the range (0,
        1). Default 0.95
    cycle_length : `int`, optional
        The number of iterations between cooling steps for the simulated
        annealing. Default 10
    p : `float`, optional
        The probability of doing a random resample rather resampling from the
        worst stratum when performing changes. Must be in the range (0, 1).
        Default 0.5
    weights : array-like[`float`] (3,), optional
        The weights for the objective function components. See
        :func:`~.clhs_objective_func`. If `None` (default), uses a weight of
        1 for all components
    progress : `bool`, optional
        If `True` (default), display a progress bar and other progress
        information
    random_state : `int` or tuple, optional
        A random state to initialize (or a seed for integer input). Not
        setting this may cause irreproducible results. See the documentation
        for :func:`numpy.random.set_state` for requirements on tuple input.
        Default `None`

    Returns
    -------
    sampled_indices : :class:`numpy.ndarray`[`int`] (``num_samples``,)
        The indices in the sample
    remaining_indices : :class:`numpy.ndarray`[`int`]
        The indices not in the sampling, which will never contain any indices
        from ``include`` (if given). This will be 1D with length
        ``num_data - num_samples`` if all indices are valid, or
        ``num_data - num_samples - np.count_nonzero(~good_mask)``
        if any indices are masked out

    Raises
    ------
    ValueError
        Raised if ``num_samples`` is longer than the number of rows in
        ``predictors``, if ``temp_decrease`` is not in (0, 1),
        if ``initial_temp`` is not in (0, 1], or if ``p`` is not in (0, 1)
    UserWarning
        Raise if ``max_iterations`` reached without convergence
    """
    if random_state is not None:
        if isinstance(random_state, int):
            np.random.seed(0)
            np.random.seed(random_state)
        else:
            np.random.set_state(random_state)
    if good_mask is None:
        good_mask = np.ones(np.squeeze(predictors).shape[0], dtype=bool)
    if objective_func_limit is None:
        objective_func_limit = -np.inf
    n_rows = np.squeeze(predictors).shape[0]
    n_obs = np.squeeze(predictors)[good_mask].shape[0]
    if num_samples >= n_obs:
        raise ValueError("Too many samples requested ({}) for predictors with"
                         " {} observations".format(num_samples, n_obs))
    data_corr = get_correlation_matrix(predictors, good_mask)
    quantiles = get_strata(predictors, num_samples, good_mask)
    if not 0 < temp_decrease < 1:
        raise ValueError("temp_decrease should be between 0 and 1 (exclusive)")
    if not 0 < initial_temp <= 1:
        raise ValueError(
            "initial_temp should be greater than 0 and no larger than 1")
    if not 0 < p < 1:
        raise ValueError("p should be between 0 and 1 (exclusive)")
    temp = initial_temp
    sample_indices, remaining_indices = get_random_samples(
        n_rows, num_samples, good_mask, include)
    x = predictors[sample_indices].copy()
    obj, obj_continuous, obj_categorical, obj_corr = clhs_objective_func(
        x, quantiles, data_corr, weights)
    current_results = dict(
        sample_indices=sample_indices.copy(),
        remaining_indices=remaining_indices.copy(),
        x=x.copy(),
        obj=obj,
        obj_continuous=obj_continuous.copy())

    def range_func(show_progress, num, **kwargs):
        if show_progress:
            import tqdm
            return tqdm.trange(num, **kwargs)
        else:
            return range(num)

    iterator = range_func(progress, max_iterations, dynamic_ncols=True)
    last_index_reached = 0
    for i in iterator:
        last_index_reached = i
        previous_results = copy.deepcopy(current_results)
        if np.random.rand() < p:
            # Random replacement
            sample_indices, remaining_indices = resample_random(
                previous_results["sample_indices"],
                previous_results["remaining_indices"],
                include)
        else:
            # Replace item from worst stratum
            sample_indices, remaining_indices = resample_worst(
                previous_results["obj_continuous"],
                previous_results["sample_indices"],
                previous_results["remaining_indices"],
                include)
        current_results["sample_indices"] = sample_indices.copy()
        current_results["remaining_indices"] = remaining_indices.copy()
        current_results["x"] = predictors[sample_indices].copy()
        obj, obj_continuous, obj_categorical, obj_corr = clhs_objective_func(
            current_results["x"], quantiles, data_corr, weights)
        current_results["obj"] = obj
        current_results["obj_continuous"] = obj_continuous
        if obj <= objective_func_limit:
            # Reached stopping criteria
            if progress:
                iterator.close()
            break
        delta_obj = obj - previous_results["obj"]
        anneal_fac = np.exp(-delta_obj / temp)
        if delta_obj > 0 and np.random.rand() >= anneal_fac:
            # Don't take the step! revert current_results
            current_results = copy.deepcopy(previous_results)
        if i % cycle_length == 0:
            # "Cool" for the annealing
            temp *= temp_decrease
    # Either hit max_iterations, or reached objective_func_limit
    if progress:
        iterator.close()
    # Print a warning if we hit max_iterations before stopping
    if last_index_reached == max_iterations - 1:
        warnings.warn("Did not reach stopping criterion {} within {}"
                      " iterations".format(objective_func_limit,
                                           max_iterations))
    return (current_results["sample_indices"],
            current_results["remaining_indices"])


if __name__ == "__main__":
    # This is specific to what I'm using this for
    import argparse
    import pathlib
    import healpy
    from astropy.table import Table


    def valid_temp_range(arg):
        try:
            value = float(arg)
        except ValueError as e:
            raise argparse.ArgumentTypeError(str(e))
        if not 0 < value <= 1:
            raise argparse.ArgumentTypeError("Invalid value {} for temp".format(
                value))
        return value


    def valid_dtemp_range(arg):
        try:
            value = float(arg)
        except ValueError as e:
            raise argparse.ArgumentTypeError(str(e))
        if not 0 < value < 1:
            raise argparse.ArgumentTypeError(
                "Invalid value {} for dtemp".format(value))
        return value


    def valid_p_range(arg):
        try:
            value = float(arg)
        except ValueError as e:
            raise argparse.ArgumentTypeError(str(e))
        if not 0 < value < 1:
            raise argparse.ArgumentTypeError("Invalid value {} for p".format(
                value))
        return value


    parser = argparse.ArgumentParser("Conditioned Latin Hypercube sampling for"
                                     " HealPix maps")
    parser.add_argument("-m", "--mask_file", type=pathlib.Path,
                        help="File containing a mask, if any, with 'False'"
                             " corresponding to bad data which should be "
                             "excluded."
                             " Otherwise, all data is assumed to be good")
    parser.add_argument("--max", type=int, default=10000, metavar="MAX_ITERS",
                        help="Maximum number of iterations")
    parser.add_argument("--limit", type=float,
                        help="The stopping condition. Default is to keep going"
                             " until MAX_ITERS is reached")
    parser.add_argument("--temp", type=valid_temp_range, metavar="INIT_TEMP",
                        default=1, help="Initial temperature for simulated"
                                        " annealing. Must be greater than 0 "
                                        "and no larger than"
                                        " 1")
    parser.add_argument("--dtemp", type=valid_dtemp_range,
                        metavar="TEMP_DECREASE", default=0.95,
                        help="Cooling factor, where the temperature after"
                             " cooling is this value times the temperature"
                             " before. Must be between 0 and 1 (exclusive)")
    parser.add_argument("-p", type=valid_p_range, default=0.5,
                        help="Probability of doing a random replacement step"
                             " rather than replacing an item from the worst "
                             "strata."
                             " Must be between 0 and 1 (exclusive)")
    parser.add_argument("-w", "--weights", type=float, nargs=3,
                        default=np.ones(3), help="Weights on the objective"
                                                 " function components, "
                                                 "as [continuous, categorical,"
                                                 " correlation]. Note that "
                                                 "the weight for categorical"
                                                 " data must be included if "
                                                 "any are given, even though"
                                                 " this is not yet implemented")
    parser.add_argument("-c", "--cycle-length", type=int, default=10,
                        help="The number of steps between cooling for the"
                             " simulated annealing")
    parser.add_argument("--progress", action="store_true",
                        help="If given, display progress info")
    parser.add_argument("--rstate", type=int,
                        help="A seed for setting the random state. If not"
                             " given, the random state is unaltered (may cause"
                             " irreproducible results)")
    parser.add_argument("num_samples", type=int,
                        help="The number of samples to be drawn, including"
                             " extrema")
    parser.add_argument("data_files", nargs="+",
                        type=pathlib.Path, help="Files for the predictors ("
                                                " independent data)")
    args = parser.parse_args()

    data = []
    n_cols = len(args.data_files)
    for data_file in args.data_files:
        data.append(healpy.read_map(data_file.as_posix, verbose=False))
    data = np.asarray(data)
    if n_cols == 1:
        data = data.squeeze()
    else:
        if data.shape[1] != n_cols:
            data = data.T

    if args.mask_file is not None:
        if "fit" in args.mask_file.suffix.to_lower():
            mask = Table.read(args.mask_file)
        else:
            mask = np.loadtxt(args.mask_file, dtype=bool)
    else:
        mask = np.ones(data.shape[0], dtype=bool)

    # Always include extreme values
    if n_cols == 1:
        always_include = np.array([data[mask].argmin(), data[mask].argmax()])
    else:
        always_include = []
        for col in data.T:
            imin = col[mask].argmin()
            if imin not in always_include:
                always_include.append(imin)
            imax = col[mask].argmax()
            if imax not in always_include:
                always_include.append(imax)
        always_include = np.asarray(always_include)

    sampled, remaining = clhs(data, args.num_samples, mask, always_include,
                              args.max, args.limit, args.temp, args.dtemp,
                              args.p, args.weights, args.cycle_length,
                              args.progress, args.rstate)
    sampled_path = pathlib.Path("sampled_indices.txt").absolute()
    remaining_path = pathlib.Path("remaining_indices.txt").absolute()
    np.savetxt(sampled_path, sampled, fmt="%d")
    print("Sampled indices saved to", sampled_path)
    np.savetxt(remaining_path, remaining, fmt="%d")
    print("Remaining indices saved to", remaining_path)
