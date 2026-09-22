# Release validation

## 0.2.1 engineering validation — 2026-09-22

- All **52** original package Python files match the paper commit
  `4075d2fe13b354de910d0cd1bb22826b94296594` byte for byte. The reference
  remains available by its full commit identifier.
- Fresh locked CPython 3.10.13 environment on Apple Silicon: **162 tests passed**.
  One expected warning comes from a deliberately short multimodal sequence.
  The core lock does not include PyTorch; this is not a validation of optional
  GPU/training configurations (the inherited PyTorch-only test returns early
  when that optional dependency is absent).
- Installed-wheel comparison against the paper commit: **251 numerical values**
  across six neural/fidelity/behavior/cross-modal CLI cases matched, with maximum
  observed absolute difference **0.0** and matching numerical keys/missing values.
  Tolerances are rtol=1e-10, atol=1e-12; these fixtures are regression evidence,
  not a guarantee of identical results for every dataset or runtime.
- Opt-in diagnostics versus the V2 sensitivity script on three finite-input
  cases (seeds 17/29/43): **15 values matched exactly**. The new API additionally
  validates shapes/support and rejects gaps for conditional state diagnostics.
- The wheel contains **60 entries**, includes the new diagnostic API, and excludes
  test modules and fixtures. The source distribution contains the test suite,
  compatibility manifest, and **14** fixture CSV/JSON files. Import and CLI help
  were tested from outside the checkout using the installed wheel.
- Hosted validation passed: **162 tests on Linux**, **162 on macOS**, and **69
  unit tests on Python 3.12** with current compatible dependencies. Both locked
  jobs also passed wheel checks and the six-case numerical comparison.
  [CI record](https://github.com/andrestrocyte/nethobench/actions/runs/35675393655).

See [reproduction instructions and scientific boundaries](docs/reproducibility_and_compatibility.md).
The default score definition and manuscript computations have not been replaced
by the optional sensitivity variants.

## 0.2.0 release validation

Validation was performed on 2026-07-28 using Python 3.10.

## Test coverage

- Initial complete included suite: 145 passed; one IBL dry-run test exposed that
  the no-op dry-run path incorrectly enforced the real 8 GB storage gate.
- Release fix: the storage assertion now runs only for a real dataset build.
  Scoring and model-evaluation functions were not changed.
- Post-fix targeted rerun: 12 passed, covering the IBL dry-run CLI, both neural
  CLI fixtures, the PSD output/weight assertions, and all temporal metric tests.
- The 145 tests already passing in the complete run do not traverse the moved
  dry-run assertion and required no code change.

## Package audit

A wheel was built from a temporary clean copy:

- artifact: `nethobench-0.2.0-py3-none-any.whl`
- SHA-256: `6efc32930c8f8ba16eb9bc4fad333d37756b6148b677e10b02a5b3eba4551329`
- wheel entries: 70
- required temporal, composite, definition, and IBL scoring modules: present
- data, generated outputs, paper, and rebuttal directories: absent

## PSD-integrated scoring smoke test

The neural CLI was run on the compact `ba4` fixture. The release emitted the
current metric names, legacy aliases, PSD audit sidecars, temporal family, and
final composite.

| Quantity | Score |
|---|---:|
| `TRJDIST_score` | 0.747469187 |
| `ACF_score` | 0.729305855 |
| `PSD_score` | 0.565724677 |
| `family_temporal_spectral` | 0.680833239 |
| `FINAL_COMPOSITE_SCORE` | 0.646776466 |

The validation script asserted to absolute tolerance \(10^{-12}\) that

\[
\texttt{family\_temporal\_spectral}
=
\frac{
\texttt{TRJDIST\_score}
+
\texttt{ACF\_score}
+
\texttt{PSD\_score}
}{3}.
\]

It also asserted that all required values were finite.
