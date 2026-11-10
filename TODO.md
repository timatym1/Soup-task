# Project TODO

Statuses are evidence-based: `TODO`, `IN_PROGRESS`, `DONE`, or `BLOCKED`. Priorities: `P0` release/safety critical, `P1` required for a credible release, `P2` improvement.

Progress is measured as completed work packages divided by all listed work packages. Packages have equal weight; blocked work is not counted as complete.

| Priority | Task | Status | Completion criteria and evidence |
|---|---|---|---|
| P0 | Audit the current pipeline, configuration, tests, reports, and Git history | DONE | Source stages, dependencies, `soup.yaml`, provenance, reports, tests, ignored artifacts, and the absence of CI workflows were inspected. Repository commit history and prior release evidence were reviewed; missing local artifacts are not treated as proof of a project defect. |
| P0 | Correct the Soup CLI aggregate release gate | DONE | Gate is `PASS` only when System Doctor, Data Doctor, Pre-flight, and Ship each have successful evidence; missing checks, failed checks, or blocked pre-flight cannot pass. Covered by `test_soup_cli_gate_passes_only_when_all_release_commands_pass`. |
| P0 | Run local regression tests and static checks | DONE | `python -m unittest discover -s tests -v` passes all 16 tests; `python -m compileall -q src tests` and `git diff --check` pass. |
| P1 | Add automated repository checks | DONE | `.github/workflows/tests.yml` runs all unit tests and Python compilation on pushes and pull requests using only Python's standard library. The same commands pass locally; hosted workflow execution awaits a GitHub event. |
| P0 | Add autonomous-work and handoff documentation | DONE | `TODO.md`, `HUMAN_REQUIRED.md`, `PROGRESS.md`, and `EXECUTION_RULES.md` exist in the repository root and agree on current blockers and next steps. |
| P1 | Validate the recorded dataset against the actual training and test files | BLOCKED | Restore the existing `data/train.jsonl` and `data/test.jsonl` from the original run; `python -m src.data_validation` must report `PASS`, and the recomputed hashes must match provenance or be explicitly reconciled. The current checkout does not contain these files. |
| P1 | Verify base, merged, and quantized model artifacts and run matched evaluation | BLOCKED | Restore existing artifacts and runtime dependencies in the CUDA Colab environment; run verification and the separate quality/safety suites on the same prompts; save complete responses and evidence. Do not retrain just to satisfy this check. |
| P0 | Complete independent quality and safety review | BLOCKED | A human reviewer assesses every generated response for every model, records the review outcome and notes in the review fields, and flags any unsafe response. Current reports contain no generated responses in this checkout. |
| P1 | Complete valid Soup CLI release evidence | BLOCKED | Resolve the CLI/project training-dependency compatibility without weakening existing pins; run Doctor, Data Doctor, Pre-flight, and Ship against restored data and real numeric evaluation evidence. The CLI must accept the documented schema, and all release gates must pass; no metrics or verdicts may be synthesized. |

`DONE` above refers only to the stated criteria. Historical reports that training, merge, or quantization completed are not re-certified by this checkout's tests.
