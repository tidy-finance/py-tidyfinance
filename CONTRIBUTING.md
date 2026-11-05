# Contributing to tidyfinance

Thank you for your interest in `tidyfinance`. This page explains how to get
help, report problems, and contribute code or documentation. By taking part,
you agree to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting help

- Read the [documentation](https://python.tidy-finance.org), including the
  [function reference](https://python.tidy-finance.org/reference/index.html),
  and the book [Tidy Finance with Python](https://www.tidy-finance.org).
- For questions about using the package, open an
  [issue](https://github.com/tidy-finance/py-tidyfinance/issues/new/choose)
  with the "Question" template. Please search existing issues first.

We maintain the package alongside our research and teaching and try to
respond within a week.

## Reporting bugs

Open an
[issue](https://github.com/tidy-finance/py-tidyfinance/issues/new/choose)
with the "Bug report" template and include:

- a minimal example that reproduces the problem, ideally with
  `domain="Pseudo Data"` so that others can run it without a WRDS
  subscription,
- the full error message or the unexpected output,
- your versions of `tidyfinance`, Python, and polars (and pandas if you use
  the pandas backend), and your operating system.

Do not post WRDS credentials, data extracts from licensed databases, or
other private information in issues.

## Suggesting features

Open an issue with the "Feature request" template. Describe the research
problem, the method or data source you want supported, and a reference if
one exists. The package mirrors the R package
[tidyfinance](https://github.com/tidy-finance/r-tidyfinance), so new
functions and arguments are usually added to both packages with the same
names and defaults.

## Contributing code or documentation

1. For anything beyond a small fix, open an issue first so that we can agree
   on the approach.
2. Fork the repository and create a branch from `main`.
3. Set up the development environment with [uv](https://docs.astral.sh/uv/):

   ```bash
   uv sync --all-extras --dev
   ```

4. Make your changes, following the conventions below.
5. Run the tests and the linter:

   ```bash
   uv run pytest
   uv run ruff check .
   uv run ruff format .
   ```

6. Add an entry to `CHANGELOG.md` under `## Unreleased`.
7. Open a pull request against `main` and link the issue. GitHub Actions
   runs the test suite on Linux, macOS, and Windows for all supported Python
   versions, and a maintainer reviews every pull request before it is
   merged.

### Conventions

- The package computes in polars. Public functions accept and return polars
  data frames, and the pandas backend converts at the package boundary (see
  `backend.py`). Do not import pandas inside modules.
- Every public function has a NumPy-style docstring with an `Examples`
  section in a fenced `python` code block (no `>>>` prompts), and a test
  file `tests/test_<name>.py`.
- Keep lines within 80 characters (Ruff).
- A function becomes public by living in a public (non-underscore) module;
  prefix internal helpers with `_`.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE).
