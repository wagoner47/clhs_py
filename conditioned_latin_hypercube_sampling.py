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
           "counts_matrix", "continuous_objective_func", "corr_objective_func",
           "clhs_objective_func", "resample_random", "resample_worst", "clhs"]
__author__ = "Erika Wagoner"
__copyright__ = "Copyright 2019, Erika Wagoner"
__credits__ = ["Erika Wagoner"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Erika Wagoner"
__email__ = "wagoner47+clhs@email.arizona.edu"
__status__ = "Development"

def get_strata(data, num_samples, mask=None):
    """
    Get the quantiles of `data` for sampling strata. This assumes that
    `num_samples` is already less than the number of (possibly masked) rows
    (the size along the 0th axis): this condition is not checked!

    :param data: Array-like of input data to stratify. Should be 1D or 2D, with\
    the size along the 0th axis being larger than `num_samples`
    :type data: array-like
    :param num_samples: The number of strata to find. This will actually find\
    `num_samples + 1` quantiles of the data. Must be at least 1!
    :type num_samples: `int`
    :param mask: A boolean array for selecting data, where `True` means the\
    data is good (included). If `None` (default), all data is assumed to be\
    good.
    :type mask: array-like of `bool`, optional
    :return: The quantiles computed (for each parameter if 2D). The shape will\
    be (`num_samples + 1`,) if 1D or (`num_samples + 1`,\
    `numpy.shape(data)[1]`) if 2D
    :rtype: :class:`numpy.ndarray`
    """
    num_samples = int(num_samples)
    if mask is None:
        mask = np.ones(np.shape(np.squeeze(data))[0], dtype=bool)
    if num_samples < 1:
        raise ValueError("Invalid number of samples: {}".format(num_samples))
    if np.ndim(np.squeeze(data)) == 1:
        return np.quantile(np.squeeze(data)[mask],
                           np.linspace(0, 1, num=(num_samples+1)))
    else:
        return np.quantile(np.squeeze(data)[mask],
                           np.linspace(0, 1, num=(num_samples+1)),
                           axis=0)

def get_correlation_matrix(data, mask=None):
    """
    Get the correlation matrix of the data. This handles the case of
    1D and 2D data as well.

    :param data: The data for which to find the correlation matrix. If 2D\
    (unless trivially 2D), rows (0th axis) should correspond to different\
    observations and columns (1st axis) should be different variables
    :type data: array-like
    :param mask: A boolean array for selecting data, where `True` means the\
    data is good (included). If `None` (default), all data is assumed to be\
    good.
    :type mask: array-like of `bool`, optional
    :return: The correlation matrix, or a scalar if the data is 1D or may be\
    squeezed to 1D
    :rtype: `float` or :class:`numpy.ndarray` of `float`
    """
    if mask is None:
        mask = np.ones(np.shape(np.squeeze(data))[0], dtype=bool)
    if np.ndim(np.squeeze(data)) == 1:
        return np.corrcoef(np.squeeze(data)[mask])
    else:
        return np.corrcoef(np.squeeze(data)[mask], rowvar=False)

def get_random_samples(num_data, num_samples, mask=None, always_include=None):
    """
    Randomly choose indices as the initialization for the cLHS. The optional
    parameter `always_include` gives an index or indices which should be
    in the selected indices, and are therefore not included in the random
    selection. They are included in the resulting sampled indices, however.

    :param num_data: The number of observations in the data (the length along\
    the 0th axis)
    :type num_data: `int`
    :param num_samples: The number of samples to select, which should be less\
    than `num_data` (or less than `num_data - len(always_include)` if\
    `always_include` is not `None`)
    :type num_samples: `int`
    :param mask: A boolean array for selecting data, where `True` means the\
    data is good (included). If `None` (default), all data is assumed to be\
    good.
    :type mask: array-like of `bool`, optional
    :param always_include: Indices that should definitely be included in the\
    sampling, if any. If a scalar, only one index is definitely included.\
    `None` (default) indicates that the sampling should be drawn from all\
    indices, so that none are definitely included.
    :type always_include: scalar or 1D array-like of `int`, optional
    :return sampled_indices: The indices that are included in the sampling\
    (including any from `always_include`)
    :rtype sampled_indices: size `num_samples` :class:`numpy.ndarray` of `int`
    :return remaining_indices: The indices remaining after selecting the\
    random sample. Any indices in `always_include` will not be in this
    :rtype remaining_indices: size `num_data - num_samples`\
    :class:`numpy.ndarray` of `int`
    """
    if mask is None:
        mask = np.ones(num_data, dtype=bool)
    masked_indices = np.arange(num_data, dtype=int)[mask]
    if always_include is None:
        always_include = []
    always_include_ = np.unique(always_include).astype(int)
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
    """
    Get eta, the number of samples in `x` binned by `quantiles` in each
    variable, for continuous variables. The shape of eta is the same as the
    shape of `x`, and the shape of `quantiles` should be
    (`numpy.shape(x)[0] + 1`, `numpy.shape(x)[1]`) for 2D, or
    (`numpy.size(x) + 1`,) for 1D

    :param x: The samples to be binned
    :type x: array-like
    :param quantiles: The bin edges with the 0th axis having size 1 greater\
    than the 0th axis of `x`. The 1st axis, if any, corresponds to the\
    different parameters
    :type quantiles: array-like
    :return eta: The count matrix, counting the number of samples in strata.\
    The shape is the same as `x`
    :rtype eta: :class:`numpy.ndarray` of `int`
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
    """
    Calculate the objective function for the counts in strata

    :param x: The samples
    :type x: array-like
    :param quantiles: The quantiles for the variable(s). See docs\
    for :func:`~.counts_matrix` for more details about shape\
    requirements
    :type quantiles: :class:`numpy.ndarray`
    :return: The values of the objective function for the strata counts over\
    each variable
    :rtype: :class:`numpy.ndarray` of `int`
    """
    eta = counts_matrix(x, quantiles)
    return np.sum(np.abs(eta - 1), axis=1)

def correlation_objective_func(data_corr, x):
    """
    Calculate the objective function for the correlation matrix of the\
    continuous variables

    :param data_corr: The correlation matrix for the full data
    :type data_corr: scalar or :class:`numpy.ndarray` of `float`
    :param x: The samples
    :type x: array-like
    :return: The value of the objective function for the correlation matrix
    :rtype: `float`
    """
    x_corr = get_correlation_matrix(x)
    return np.sum(np.abs(data_corr - x_corr))

def clhs_objective_func(x, quantiles, data_corr, weights=None):
    """
    Get the value of the full objective function. Note that at this time,
    including categorical data has not yet been implemented, but should be
    included in the future (for, e.g., star-galaxy separation).

    :param x: The samples
    :type x: array-like
    :param quantiles: The quantiles of the variable(s). See the docs for\
    :func:`~.counts_matrix` for details about shape requirements
    :type quantiles: :class:`numpy.ndarray`
    :param data_corr: The correlation matrix (or scalar for 1D) of the full\
    input data
    :type data_corr: scalar or :class:`numpy.ndarray` of `float`
    :param weights: If not `None`, the weights to assign to each objective\
    function, ordered as (1) continuous variables, (2) categorical variables,\
    (3) correlation matrix. Note that this should have 3 entries even though\
    categorical variables are not yet implemented. Default `None` uses a weight\
    of 1 for all
    :type weights: length 3 array-like, optional
    :return: The total objective function value for the sample
    :rtype: `float`
    """
    if weights is not None:
        weights_ = np.asarray(weights)
    else:
        weights_ = np.ones(3)
    obj_continuous = continuous_objective_func(x, quantiles)
    obj_categorical = 0
    obj_corr = correlation_objective_function(data_corr, x)
    obj_all = np.array([
        np.sum(obj_continuous), np.sum(obj_categorical), obj_corr])
    return np.sum(weights_ * obj_all), obj_continuous, obj_categorical, obj_corr

def resample_random(sampled_indices, remaining_indices, always_include=None):
    """
    Do random replacement of one item in the sample, but don't remove any
    indices in `always_include`

    :param sampled_indices: The indices that are part of the previous sample
    :type sampled_indices: :class:`numpy.ndarray` of `int`
    :param remaining_indices: The indices that are not part of the previous\
    sample
    :type remaining_indices: :class:`numpy.ndarray` of `int`
    :param always_include: If not `None`, these indices should always be in\
    the sample. Default `None`
    :type always_include: scalar or array-like of `int`
    :return new_sampled_indices: Resampled indices
    :rtype new_sampled_indices: :class:`numpy.ndarray` of `int`
    :return new_remaining_indices: Remaining indices after resampling
    :rtype new_remaining_indices: :class:`numpy.ndarray` of `int`
    """
    new_remaining_indices = remaining_indices.copy()
    if always_include is None:
        always_include = []
    always_include_ = np.unique(always_include)
    num_always_include = always_include_.size
    allowed_to_remove_indices = sampled_indices[
        np.isin(sampled_indices, always_include_, assume_unique=True,
                invert=True)].copy()
    idx_removed = np.random.choice(allowed_to_remove_indices.size)
    idx_added = np.random.choice(new_remaining_indices.size)
    sampled_index_to_remove = allowed_to_remove_indices[idx_removed]
    remained_index_to_sample = new_removed_indices[idx_added]
    allowed_to_remove_indices[idx_removed] = remained_index_to_sample
    new_remaining_indices[idx_added] = sampled_index_to_remove
    new_sampled_indices = np.append(allowed_to_remove_indices, always_include_)
    return new_sampled_indices, new_remaining_indices

def resample_worst(continuous_objective_values, sampled_indices,
                   remaining_indices, always_include=None):
    """
    Remove a sampled item from the worst stratum (or one of the worst strata)
    and replace it with a random item from the unsampled items. But don't
    replace anything in `always_include`

    :param continuous_objective_values: The absolute value of the count matrix\
    minus 1 summed over all parameters
    :type continuous_objective_values: array-like
    :param sampled_indices: The indices that are part of the previous sample
    :type sampled_indices: :class:`numpy.ndarray` of `int`
    :param remaining_indices: The indices that are not part of the previous\
    sample
    :type remaining_indices: :class:`numpy.ndarray` of `int`
    :param always_include: If not `None`, these indices should always be in\
    the sample. Default `None`
    :type always_include: scalar or array-like of `int`
    :return new_sampled_indices: Resampled indices
    :rtype new_sampled_indices: :class:`numpy.ndarray` of `int`
    :return new_remaining_indices: Remaining indices after resampling
    :rtype new_remaining_indices: :class:`numpy.ndarray` of `int`
    """
    new_remaining_indices = remaining_indices.copy()
    if always_include is None:
        always_include = []
    always_include_ = np.unique(always_include)
    num_always_include = always_include_.size
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

def clhs(data, num_samples, mask=None, always_include=None,
         max_iterations=10000, objective_func_limit=None,
         initial_temp=1, temp_decrease=0.95, p=0.5,
         weights=None, cycle_length=10, progress=True, random_state=None):
    """
    The full conditioned Latin Hypercube sampler. A progress bar may optionally
    be displayed, but may not be informative for large maximum number of
    iterations.

    :param data: The full data from which to sample, with the 0th axis\
    corresponding to observations and the 1st axis corresponding to individual\
    predictors (if 2D)
    :type data: array-like
    :param num_samples: The number of cLHS samples to draw from the data. This\
    must be less than the size of the 0th axis of (possibly masked) `data`
    :type num_samples: `int`
    :param mask: A boolean mask that is `True` for good rows. If `None`\
    (default), the mask is set to `True` for all rows
    :type mask: array-like of `bool`, optional
    :param always_include: If given, specifies indices that should always be\
    included in the sample. This can be an array-like or scalar.
    :type always_include: scalar or array-like of `int`
    :param max_iterations: The maximum number of iterations to try. If\
    convergence is not reached within this, a warning will be issued. Default\
    10000
    :type max_iterations: `int`, optional
    :param objective_func_limit: The stopping criteria for the objective\
    function. If `None` (default), this is set to negative infinity, so that\
    the code only stops after `max_iterations` steps
    :type objective_func_limit: `float`, optional
    :param initial_temp: An initial temperature for annealing, which should be
    between 0 and 1. Default 1
    :type initial_temp: `float`, optional
    :param temp_decrease: The cooling parameter, which must be between 0 and 1.\
    At each cooling step, the temperature is decreased to this factor of\
    itself (so it is cooled by a factor of 1 - `temp_decrease`). Default 0.95
    :type temp_decrease: `float`, optional
    :param p: The probability of doing a random replacement versus replacing\
    the worst samples for each update step. This must be between 0 and 1. A\
    value closer to 0 is more likely to replace the worst samples, while a\
    value closer to 1 is more likely to do random replacement. Default 0.5
    :type p: `float`, optional
    :param weights: The weights for combining the objective functions into an\
    overall objective function. If given, this must be of size 3 with the order\
    (1) continuous data, (2) discrete data, (3) correlation matrix. The weight\
    for discrete data must be given even though it is not yet implemented.\
    Default `None` sets all weights to 1, which is typical for most general\
    applications
    :type weights: size 3 array-like, optional
    :param cycle_length: The number of iterations between cooling for\
    annealing. Default 10
    :type cycle_length: `int`, optional
    :param progress: If `True` (default), display progress info
    :type progress: `bool`, optional
    :param random_state: A seed (if `int`) or random state (if `tuple`) to use\
    with :module:`numpy.random` (for reproducability). Default `None`
    :type random_state: `int` or `tuple`, optional
    :return sample_indices: The indices for the rows included in the sample
    :rtype sample_indices: :class:`numpy.ndarray` of `int`
    :return remaining_indices: The indices for the remaining rows not in the\
    sample, with size `numpy.shape(data)[0] - num_samples`
    :rtype remaining_indices: :class:`numpy.ndarray` of `int`
    """
    if random_state is not None:
        if isinstance(random_state, int):
            np.random.seed(0)
            np.random.seed(random_state)
        else:
            np.random.set_state(random_state)
    if mask is None:
        mask = np.ones(np.squeeze(data).shape[0], dtype=bool)
    if objective_func_limit is None:
        objective_func_limit = -np.inf
    if progress:
        import tqdm
    n_rows = np.squeeze(data).shape[0]
    n_obs = np.squeeze(data)[mask].shape[0]
    if num_samples >= n_obs:
        raise ValueError("Too many samples requested ({}) for data with"
                         " {} observations".format(num_samples, n_obs))
    data_corr = get_correlation_matrix(data, mask)
    quantiles = get_strata(data, num_samples, mask)
    if temp_decrease >= 1 or temp_decrease <= 0:
        raise ValueError("temp_decrease should be between 0 and 1")
    if inital_temp > 1 or initial_temp <= 0:
        raise ValueError("initial_temp should be beween 0 and 1")
    temp = initial_temp
    sample_indices, remaining_indices = get_random_samples(
        n_rows, num_samples, mask, always_include)
    x = data[sample_indices].copy()
    obj, obj_continuous, obj_categorical, obj_corr = clhs_objective_func(
        data_corr, x)
    current_results = dict(
        sample_indices=sample_indices.copy(),
        remaining_indices=remaining_indices.copy(),
        x=x.copy(),
        obj=obj,
        obj_continuous=obj_continuous.copy())
    def range_func(progress, max_iterations, **kwargs):
        if progress:
            return tqdm.trange(num, **kwargs)
        else:
            return range(num)
    iterator = range_func(progress, max_iterations, dynamic_ncols=True)
    for i in iterator:
        previous_results = copy.deepcopy(current_results)
        if np.random.rand() < p:
            # Random replacement
            sample_indices, remaining_indices = resample_random(
                previous_results["sample_indices"],
                previous_results["remaining_indices"],
                always_include)
        else:
            # Replace item from worst stratum
            sample_indices, remaining_indices = resample_worst(
                previous_results["obj_continuous"],
                previous_results["sample_indices"],
                previous_results["remaining_indices"],
                always_include)
        current_results["sample_indices"] = sample_indices.copy()
        current_results["remaining_indices"] = remaining_indices.copy()
        current_results["x"] = data[sample_indices].copy()
        obj, obj_continuous, obj_categorical, obj_corr \
            = clhs_objective_func(data_corr, current_results["x"])
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
    if i == max_iterations - 1:
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
                        " corresponding to bad data which should be excluded."
                        " Otherwise, all data is assumed to be good")
    parser.add_argument("--max", type=int, default=10000, metavar="MAX_ITERS",
                        help="Maximum number of iterations")
    parser.add_argument("--limit", type=float,
                        help="The stopping condition. Default is to keep going"
                        " until MAX_ITERS is reached")
    parser.add_argument("--temp", type=valid_temp_range, metavar="INIT_TEMP",
                        default=1, help="Initial temperature for simulated"
                        " annealing. Must be greater than 0 and no larger than"
                        " 1")
    parser.add_argument("--dtemp", type=valid_dtemp_range,
                        metavar="TEMP_DECREASE", default=0.95, help="Cooling"
                        " factor, where the temperature after cooling is this"
                        " value times the temperature before. Must be between 0"
                        " and 1 (exclusive)")
    parser.add_argument("-p", type=valid_p_range, default=0.5,
                        help="Probability of doing a random replacement step"
                        " rather than replacing an item from the worst strata."
                        " Must be between 0 and 1 (exclusive)")
    parser.add_argument("-w", "--weights", type=float, nargs=3,
                        default=np.ones(3), help="Weights on the objective"
                        " function components, as [continuous, categorical,"
                        " correlation]. Note that the weight for categorical"
                        " data must be included if any are given, even though"
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

    sampled, remaining = cflhs(data, args.num_samples, mask, always_include,
                               args.max, args.limit, args.temp, args.dtemp,
                               args.p, args.weights, args.cycle_length,
                               args.progress, args.rstate)
    sampled_path = pathlib.Path("sampled_indices.txt").absolute()
    remaining_path = pathlib.Path("remaining_indices.txt").absolute()
    np.savetxt(sampled_path, sampled, fmt="%d")
    print("Sampled indices saved to", sampled_path)
    np.savetxt(remaining_path, remaining, fmt="%d")
    print("Remaining indices saved to", remaining_path)
