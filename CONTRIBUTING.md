# Contributing

Thanks for helping make RBF kernels easier to understand. Fixes to confusing explanations are just as welcome as code.

## Set up

```bash
git clone https://github.com/lunganlungpg/rbf-kernel-.git
cd rbf-kernel-
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Before you open a pull request

```bash
ruff check . && ruff format --check .   # lint and formatting
pytest rbf_kernel tests README.md       # unit tests + doctests (including README code)
```

## Guidelines

- **Readable over clever.** The package is a teaching tool. Prefer clear NumPy with a comment explaining the maths.
- **Test against a reference.** If scikit-learn has an equivalent, assert that both agree (see `tests/test_kernels.py`).
- **Report numbers you measured.** Any result quoted in the docs should come from a script in `examples/` that anyone can rerun.
- **Website changes** live in `site/`. It is plain HTML, CSS and JavaScript with no build step. Open `site/index.html` in a browser, and check light mode, dark mode and phone width.
