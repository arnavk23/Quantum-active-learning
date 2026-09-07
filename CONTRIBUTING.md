# Contributing

Thanks for your interest in this project. It is small and maintained by one
person, so please keep expectations calibrated accordingly, but contributions
are welcome.

## Reporting issues

Please open a GitHub issue for bugs, unexpected results, or documentation
gaps. Useful bug reports include:

- What you ran (exact command / script)
- What you expected vs. what happened
- Your Python version and `pip freeze` output for the relevant packages
- Whether you can reproduce it with the test suite (`pytest tests/`)

## Development setup

```bash
git clone https://github.com/arnavk23/quantum_al.git
cd quantum_al
python -m venv .venv
source .venv/bin/activate        # or .venv\Scripts\activate on Windows
pip install -e ".[test,circuit]"
pytest tests/ -v
```

## Making changes

- Keep `src/quantum_al/` for the core, tested formalism and baselines.
  Anything added here should have a corresponding test in `tests/`.
- Runnable experiment/analysis scripts go in `benchmarks/`.
- If you change anything in `src/quantum_al/operator.py`, rerun
  `python -c "from quantum_al.operator import self_test; self_test()"`
  and make sure it still passes; that check is what guarantees the
  formalism matches the classical-limit proof in the papers.
- Please do not report a new result from `benchmarks/` scripts by hand-editing
  numbers into `results/` or the papers. If a script fails, fix it or report
  the failure; this project exists specifically because an earlier version
  did not follow that rule. See `results/SUMMARY.md` for the full story.

## Pull requests

- One logical change per PR where practical.
- Run `pytest tests/ -v` before opening the PR; CI will also run it.
- Describe what you changed and why in the PR description.

## Code of conduct

Be respectful and constructive. Disagreements about results or methodology
are fine and expected; bad-faith conduct is not.
