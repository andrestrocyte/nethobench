# Release validation

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
