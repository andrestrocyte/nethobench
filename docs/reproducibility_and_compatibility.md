# Reproducibility and compatibility

## Published scores

The ICLR V2 manuscript pins commit
`4075d2fe13b354de910d0cd1bb22826b94296594` (package 0.2.0).
The engineering release 0.2.1 preserves all 52 Python source files present in
that commit byte for byte. The default API, CLI, metric names, legacy aliases,
family membership, weights, missing-component policy and scoring formulas are
unchanged. Package version and opt-in diagnostics are separate from the score
definition. Renaming a GitHub repository does not change its commit identities.

`tests/resources/published_source_hashes.json` records the original hashes.
The corresponding test prevents an unnoticed edit to the pinned implementation.
A future intentional scoring change needs a separately versioned definition and
new validation; do not update this manifest merely to make a failing test pass.

## Locked scoring and test environment

Use CPython 3.10.13. `requirements/py310.lock` pins the complete core scoring,
test and build dependency graph, with artifact hashes and platform markers.
The numerical-library constraints match the V2 audit environment. This is a
core CPU-scoring environment, not a complete GPU-training or manuscript-build
environment. Optional PyTorch training is outside this lock.

```bash
python -m pip install uv==0.11.28
uv venv .venv --python 3.10.13
uv pip install --python .venv/bin/python --require-hashes -r requirements/bootstrap.lock
uv pip sync --python .venv/bin/python --no-build-isolation --require-hashes requirements/py310.lock
uv pip install --python .venv/bin/python --no-deps --no-build-isolation -e .
MPLBACKEND=Agg NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m pytest -q
```

The bootstrap lock fixes the build tools before any source dependency is built.
Normal `pip install .` remains supported and retains the original dependency
lower bounds; it does not promise the same numerical environment as the lock.
OS, architecture, BLAS, threading and hardware can still affect floating-point
results. Hash locking alone is not a promise of identical results on all systems.

Regenerate intentionally, review the dependency diff, then rerun compatibility:

```bash
uv pip compile requirements/build.in --generate-hashes --python-version 3.10 --universal -o requirements/bootstrap.lock
uv pip compile pyproject.toml requirements/build.in --extra test --constraint requirements/scoring-constraints.txt --generate-hashes --python-version 3.10 --universal -o requirements/py310.lock
```

## Automated verification

GitHub Actions runs the locked unit/integration suite on Linux and macOS,
builds a wheel and source distribution, checks that the wheel excludes tests,
and imports/runs the installed CLI outside the checkout. A separate Python 3.12
unit-test job checks installation with current compatible dependencies.

`scripts/compare_published_release.py` compares six CLI cases against the actual
paper commit in the same environment: neural and fidelity scoring for two
fixtures, behavioral scoring and cross-modal scoring. It checks numerical key
sets, missing values and every numerical leaf (relative tolerance 1e-10,
absolute tolerance 1e-12). This is regression evidence, not proof for every
possible dataset. The archived baseline is read from local Git history, so use
a full clone or fetch the paper commit before running it.

Small integration fixtures are restored unchanged from public `nethobench`
commit `652d931a15a1036647227259efd65de2cc7e4da6`. They are included with tests in
the source distribution, but neither tests nor fixtures are installed by the
wheel. The README examples use a source checkout.

## Opt-in implementation sensitivity

```python
from nethobench.diagnostics.implementation_sensitivity import (
    compute_relational_sensitivity,
    compute_conditional_state_sensitivity,
)

# Aligned arrays with shape (sequence, time, channel).
relational_checks = compute_relational_sensitivity(reference, prediction)
state_checks = compute_conditional_state_sensitivity(reference, prediction)
```

These reproduce the finite-input calculations of the V2 sensitivity analysis,
with explicit validation and support reporting. Results are separate diagnostic
dictionaries under protocol `sequence-directed-conditional-v1`, not replacements
for default score keys. They are never automatically added to a composite.
Record sample frequency and lag units with exported diagnostics.

The relational variant pairs samples within each sequence before filtering
missing endpoints and compares all directed off-diagonal matrix entries. Lag
covariance uses lags 1, 2, 4 samples; VAR(1) uses ridge 0.01. Reference and
prediction retain the separately fitted robust scaling of the sensitivity
protocol. VAR agreement remains a statistical proxy, not a causal estimate.

The state variant retains reference-fitted PCA and K=11 clustering but compares
row-conditional transition distributions, uniformly averaging reference-supported
origin states. A missing predicted origin contributes zero; no reference support
is undefined. This separates conditional transitions from occupancy. Finite
tensors are required to avoid constructing transitions across missing samples.

## Scientific boundaries

The published lag/VAR definition pools sequences and compares one matrix
triangle. Its transition score compares normalized joint transition counts,
which combine occupancy with conditional dynamics. These remain the default
definitions to reproduce the paper. Use the new explicit variants for sensitivity
analysis and rescore every compared model on matched support before interpreting
an alternative ranking. Do not mix default and alternative scores in one table.

Neither variant establishes biological states, causality, conditional predictive
validity, preservation of all low-variance information, or downstream utility.
Those require additional experimental endpoints. The main widefield uncertainty
and human preprocessing limitations in the manuscript cannot be fixed by a
software engineering release. Existing manuscript figures and results are not
rewritten or recomputed by this change.
