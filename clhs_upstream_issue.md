# cLHS upstream issue and PR handoff

## Implementation and validation on 8 November 2026

The local implementation fixes the three indexing defects and the mismatch
between stratum objectives and observation scores. It also corrects the
categorical objective when a sample omits a category and supports swaps for
inputs containing only categorical predictors. The original reproduction
notes below describe the unfixed release and checkout at commit
`01a957584a5bec0d25309b2bff6fda9038284275`.

The sampling fix is commit `6994cb0`. Packaging and compatibility work follows
in a separate commit. No upstream issue, pull request, or release has been
submitted.

The exported resampling helpers retain their signatures and parameter names.
Their `continuous_objective_values` argument now explicitly requires scores
aligned with `sampled_indices`. For each observation, the sampler sums the
count minus one in its occupied bin across continuous predictors. The
stratum objectives returned by `continuous_objective_func`,
`clhs_objective_func`, and `clhs` retain their existing meaning. Protected
observations contribute to bin counts but cannot be removed. Bin assignments
follow NumPy histogram boundaries, including repeated edges and the final
endpoint. Scores are recomputed from the accepted sample before each proposal.
Corrected scoring changes some sampled results; repeating a seed within an
environment remains reproducible.

Integer columns remain categorical. Convert integer measurements to float
for continuous treatment. Inputs containing only categorical predictors use
equal observation scores, so both proposal branches choose uniformly among
removable samples.
The categorical objective still controls proposal acceptance.

The package now declares its setuptools backend and metadata in
`pyproject.toml`, with Python `>=3.9`, NumPy `>=1.26.0`, pandas `>=2.0.0`, and
tqdm `>=4.66.0`. These are tested support floors, not claims that every older
version fails. No dependency upper bounds were needed. A version module with
no dependencies supplies both the distribution and module version, currently
`1.0.2`. A future release must choose a new release number. Build and test
requirements are separate from runtime requirements. Installation instructions
now use pip or an optional uv environment.

### Results on macOS with Apple Silicon

All 44 tests passed in every combination below. Each run imported the built
wheel from a temporary environment outside the checkout and treated warnings
as errors.

| Python | NumPy | pandas | tqdm | pytest | Result |
|---|---|---|---|---|---|
| 3.9.6 | 1.26.0 | 2.0.0 | 4.66.0 | 8.4.2 | 44 passed |
| 3.9.6 | 2.0.2 | 2.3.3 | 4.70.1 | 8.4.2 | 44 passed |
| 3.10.22 | 2.2.6 | 2.3.3 | 4.70.1 | 9.2.0 | 44 passed |
| 3.11.17 | 2.4.6 | 3.1.1 | 4.70.1 | 9.2.0 | 44 passed |
| 3.12.3 | 2.5.4 | 3.0.6 | 4.70.1 | 9.2.0 | 44 passed |
| 3.12.3 | 2.5.4 | 3.1.1 | 4.70.1 | 9.2.0 | 44 passed |
| 3.13.17 | 2.5.4 | 3.1.1 | 4.70.1 | 9.2.0 | 44 passed |
| 3.14.9 | 2.5.4 | 3.1.1 | 4.70.1 | 9.2.0 | 44 passed |
| 3.15.0 | 2.5.4 | 3.1.1 | 4.70.1 | 9.2.0 | 44 passed |

The editable Python 3.12 installation and a clean installation of the source
distribution also passed all 44 tests. An isolated build with setuptools
84.2.1 produced both distribution formats. Python 3.9 with the declared minimum
setuptools 77.0.3 also built both formats, with warnings treated as errors.
Both primary artifacts passed `python -m twine check --strict`. The wheel
contains only the package and distribution metadata; the source distribution
includes the tests.

The regression suite covers the original examples, observation mapping,
ties, protected samples, input preservation, partition invariants, invalid
helper inputs, repeated bin edges, dtype handling, missing categories,
masked observations, both progress settings, seed reproducibility, accepted
sample scoring after rejected proposals, syntax warnings, and package metadata.
The original targeted tests failed before the fix, with 10 failures and one
pass. The checkout already used raw docstrings, so the published release's
escape warnings required no further docstring changes.

GitHub Actions now defines the same dependency matrix and clean wheel/source
installation checks on Linux. Those hosted jobs have not run locally.

To use the local development environment:

```sh
uv venv --python 3.12
uv pip install -e ".[test]"
uv run --no-project python -m pytest
```

The local `.venv`, build artifacts, and package metadata directories are
ignored by Git. Additional test runtimes and dependency caches are under
`/private/tmp`.

### Remaining work outside this change

An extra probe at a low temperature with `initial_temp=0.001` exposed an existing
overflow in the annealing loop's exponential calculation for improving
proposals. The compatibility results above use the default temperature and
100 iterations. This change does not revise the annealing schedule or claim
that all temperature/iteration combinations are free of numerical warnings.

The course project's compatibility wrapper and grouped split checks still
need validation against a future fixed release. The two existing upstream
reports below have not been established as resolved by this change.

## Original investigation

Suggested title: **Correct resampling observation scores and add compatibility tests**

Target repository: [wagoner47/clhs_py](https://github.com/wagoner47/clhs_py).
Prepared on 8 November 2026. This document is a local handoff; no upstream issue or PR has been submitted.

## Problem and scope

The published Python package `clhs==1.0.2` fails during sampling with NumPy 2.5.4. Investigation found three problems in `resample_worst`:

- A single eligible worst position can remain an array, causing a scalar assignment error.
- The function confuses objective ranks with sample positions, so tied scores can cause it to remove the wrong sample.
- With `include`, it can return the same index in both selected and remaining arrays and lose another index entirely.

Fix the indexing and add regression tests. Review dependency declarations and establish a tested compatibility range. Changing dependency bounds alone will not fix the incorrect selection or overlapping sets.

The function is in [`clhs/clhs.py`](https://github.com/wagoner47/clhs_py/blob/master/clhs/clhs.py). Compare the current GitHub checkout with the published release before editing; their formatting and line numbers differ.

## Verified environment

| Component | Observed version |
|---|---|
| Python | 3.12.3 |
| Installed distribution | clhs 1.0.2 |
| `clhs.__version__` | 1.0.1 |
| NumPy | 2.5.4 |
| pandas | 3.0.6 |
| tqdm | 4.70.1 |

These are reproduction versions, not proposed minimum requirements. No compatibility matrix has been run yet.

The published distribution declares `tqdm`, `numpy`, and `pandas` without version bounds. The upstream [`setup.py`](https://github.com/wagoner47/clhs_py/blob/master/setup.py) also lacks `python_requires` and reports version 1.0.2. The module version mismatch is independently reproducible with:

```python
from importlib.metadata import version
import clhs

print(version("clhs"))  # 1.0.2
print(clhs.__version__)  # 1.0.1
```

## Reproduce without the wave energy data

Use a Python 3.12 environment with these installed packages:

```sh
python -m pip install "clhs==1.0.2" "numpy==2.5.4" "pandas==3.0.6" "tqdm==4.70.1"
```

### 1. Single maximum raises an exception

```python
import numpy as np
import clhs

np.random.seed(42)
clhs.resample_worst(
    np.array([0.0, 9.0, 1.0]),
    np.array([30, 10, 20]),
    np.array([40, 50]),
)
```

Observed error:

```text
TypeError: only 0-dimensional arrays can be converted to Python scalars

The above exception was the direct cause of the following exception:

ValueError: setting an array element with a sequence.
```

The final failure is the assignment to `new_remaining_indices[idx_added]` inside `resample_worst`.

Expected behavior: exchange one selected index with one remaining index, returning disjoint arrays that preserve the original combined pool. Under the function's current indexing convention, score 9 belongs to position 1, whose selected index is 10.

### 2. Tied maxima remove a lower-scoring position

```python
import numpy as np
import clhs

np.random.seed(42)
selected, remaining = clhs.resample_worst(
    np.array([0.0, 0.0, 9.0, 9.0]),
    np.array([30, 10, 20, 50]),
    np.array([40, 60]),
)
print(selected.tolist())
print(remaining.tolist())
```

Observed output:

```text
[30, 60, 20, 50]
[40, 10]
```

The function removes index 10 at position 1, which has score 0. The largest scores are at positions 2 and 3. Under the current convention, a replacement should remove index 20 or 50.

### 3. `include` can produce overlapping sets

```python
import numpy as np
import clhs

np.random.seed(42)
selected, remaining = clhs.resample_worst(
    np.array([0.0, 0.0, 9.0, 9.0]),
    np.array([30, 10, 20, 50]),
    np.array([40, 60]),
    include=[30],
)
print(selected.tolist())
print(remaining.tolist())
```

Observed output:

```text
[30, 10, 60, 50]
[40, 50]
```

Index 50 appears in both arrays and index 20 disappears from the combined pool. This violates sampling without replacement even though the protected index 30 remains selected.

### 4. The public sampler also fails

This fixture needs no external files. Its generated inputs are only for reproducing the package failure.

```python
import numpy as np
import pandas as pd
import clhs

inputs = pd.DataFrame(
    np.random.default_rng(0).random((20, 3)),
    columns=["x1", "x2", "x3"],
)
clhs.clhs(
    inputs,
    num_samples=10,
    max_iterations=100,
    random_state=42,
    progress=False,
)
```

This raises the same scalar assignment error through `clhs` → `resample` → `resample_worst`.

## Diagnosis

Keep three different quantities separate: indices into the original dataset, positions within the selected array, and ranks of distinct objective values.

For scores `[0, 0, 9, 9]`, `np.unique(..., return_inverse=True)` produces inverse ranks `[0, 0, 1, 1]`. Filtering for the highest rank leaves `[1, 1]`, not the positions `[2, 3]` of the highest scores. The existing function uses those rank values as positions.

When only one position remains, the conditional random choice does not run. The position stays a one-element array and the replaced sample also stays an array. NumPy rejects assigning that array to a scalar slot. NumPy documented deprecation of this scalar coercion in its [1.25 release notes](https://numpy.org/doc/2.0/release/1.25.0-notes.html#deprecations). The exact first NumPy version that raises an error has not been established here.

The function also applies the eligible-position lookup twice when retrieving the removed sample. Once `include` makes eligible positions nonconsecutive, this can retrieve a different sample from the one being replaced. That explains the overlapping sets in reproduction 3.

## Local workaround and an algorithm question to resolve

The course project currently uses [clhs_compat.py](clhs_compat.py). Its replacement function is self-contained below so this handoff can be moved to another repository:

```python
def resample_worst(objective, selected, remaining, include=None):
    selected = selected.copy()
    remaining = remaining.copy()
    eligible = np.flatnonzero(
        ~np.isin(selected, [] if include is None else include)
    )
    scores = np.asarray(objective)[eligible]
    worst = int(np.random.choice(eligible[scores == scores.max()]))
    added = int(np.random.choice(len(remaining)))
    selected[worst], remaining[added] = remaining[added], selected[worst]
    return selected, remaining
```

This requires `import numpy as np`. It selects scalar positions, handles ties, preserves protected samples, and swaps the actual removed index into the remaining array. The wrapper temporarily replaces the installed function and restores it afterward. An upstream fix should edit the package directly and retain its public parameter names.

**Review the meaning of the objective array before treating this as a complete algorithm correction.** The workaround treats objective entries as scores for selected positions, following the existing function's indexing convention. However, `continuous_objective_func` describes its output as contributions by stratum. Confirm how a worst stratum should map to actual selected observations. Fixing scalar/index handling does not establish that this mapping implements the intended cLHS method. Add a test for the agreed mapping, or document and separate that algorithm change.

Define behavior for empty remaining arrays or no eligible positions. The public sampler normally prevents these cases, but direct helper calls should have a clear error rather than an obscure NumPy failure.

## Dependency and packaging work

The proposed dependency changes should follow tests, not copy the course project's version pins.

1. Choose supported Python versions and declare `requires-python` or `python_requires`. Python 3.12 is the only version verified here.
2. Test the fix against NumPy 2 and a compatible NumPy 1.26 environment if retaining NumPy 1 support. Treat NumPy 1.25's deprecation behavior as a separate test environment if useful; it does not support Python 3.12.
3. Test pandas 2 and pandas 3 on compatible Python/NumPy combinations. No pandas failure has been isolated in this investigation, so do not claim pandas is the cause of the observed crash.
4. Derive lower dependency bounds from the oldest combinations that pass. Add an upper bound only for a demonstrated incompatibility. Do not use `numpy<2` as the sole fix; that leaves the indexing errors intact.
5. Add an explicit build backend and build requirements in `pyproject.toml`, following the [Python Packaging User Guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/). Runtime dependencies belong separately from build and test requirements.
6. Check float32, float64, and integer input handling. The installed release classifies continuous columns using `select_dtypes(include=float)`. Our notebook explicitly converts predictors to float; determine and document the intended behavior for other numeric dtypes.
7. Align distribution and module versions. Keep one authoritative version source or test that both agree.
8. Check a freshly built wheel and source distribution in clean environments. Update the installation instructions to use pip and the declared build backend.

Fresh compilation of the installed 1.0.2 source under Python 3.12 also emitted four `SyntaxWarning` messages for invalid docstring escapes, at installed-source lines 205, 252, 300, and 343. Use raw docstrings or escaped backslashes where needed. Some GitHub master docstrings already use raw strings, so check whether this requires code changes or only a new release.

Do not introduce a new random-number API in the same bug fix unless needed. The package uses NumPy's global random state; our local wrapper restores that state for notebook isolation. Changing upstream random-state behavior can be a separate improvement.

## Regression tests and acceptance criteria

- Reproduce the singleton crash before the fix and verify it no longer raises afterward.
- Verify a chosen worst position follows the agreed score/stratum interpretation, including ties away from the start of the selected array.
- Preserve all `include` indices, including nonconsecutive protected positions and the case with only one removable position.
- After every swap, preserve lengths, uniqueness, disjoint selected/remaining sets, and their combined pool. Do not modify input arrays.
- Run the public sampler fixture above with `progress=False` and check finite objectives and valid selected/remaining indices.
- Repeating a call with the same `random_state` should reproduce its selected indices.
- Test tied predictor levels, missing categories, and masked observations separately. Repeated quantile edges should not be assumed to be the cause of this indexing error.
- Run deprecation and syntax checks without globally suppressing warnings. Exercise both progress settings and both distribution formats.
- Record the dependency matrix that actually passes before declaring support in package metadata.

For reference, the local workaround passed scalar/index regression checks and sampled 50 distinct groups from 61 candidate groups in our measured dataset. Restoring repeated observations produced 57 training rows, 12 validation rows, and the unchanged 11-row test set. Checks covered complete row preservation, group separation, reproducibility, and restoring the package function and NumPy random state after success and failure. This validates the local integration, not every package feature or dependency combination.

## Related upstream reports and completion

The existing [NumPy error report, issue 11](https://github.com/wagoner47/clhs_py/issues/11), involves the same resampling area but a different exception. [Issue 9](https://github.com/wagoner47/clhs_py/issues/9) concerns quantile bins and repeated values. Neither report alone establishes that it is the same bug; reproduce before linking them as resolved.

Start with the indexing fix and its tests. Dependency metadata and packaging can be a second commit or PR if that makes review easier. Capture the upstream commit used and include actual test results in the PR description.

After a fixed release is available, update the course project's dependency and replace `sample_clhs` with the upstream `clhs.clhs` call. Remove the local compatibility helper only after rerunning the grouped split checks.
