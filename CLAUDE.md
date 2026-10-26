# CLAUDE.md

Guidance for working in this repository.

## Project layout

- `tidyfinance/` — the package. **Internals are implemented in polars**:
  module-level functions take and return `polars.DataFrame` objects, and
  calendar-date columns are `polars.Date`. The public API additionally
  accepts pandas input and returns pandas under the `pandas` backend (see
  `backend.py`; the default backend is `polars`).
  - `__init__.py` — builds the public API automatically by scanning the
    public submodules and re-exporting their functions/classes (see
    `__all__`). Data-bearing functions are wrapped at this boundary by
    `backend._use_backend` so they honor the active pandas/polars backend
    (default `polars`: a pass-through; the `pandas` backend converts
    pandas in → polars internals → pandas out). Keep this file's namespace clean:
    discovery loop variables, imports (`importlib`, `pkgutil`, `types`),
    and internal toggles are deleted or underscore-prefixed at the end so
    they don't leak into `dir(tidyfinance)` or the docs.
  - `backend.py` — `set_backend` / `get_backend`, the `_use_backend`
    decorator, and the `_DATE_COLUMNS` list of calendar-date columns.
  - `lagging.py`, `portfolios.py`, `regression.py` — analytics functions
    (lagged joins, portfolio sorts/breakpoints, beta / Fama-MacBeth
    estimation).
  - `download.py` — the `download_data` dispatcher;
    `download_open_source.py`, `download_tidy_finance.py`,
    `download_wrds.py`, `download_pseudo.py` — per-source download
    helpers. SQL goes through `download_wrds._read_sql`
    (`polars.read_database`); HTTP payloads are parsed with
    `polars.read_csv` / `read_parquet` on fetched bytes.
  - `utilities.py`, `supported_datasets.py` — helpers and dataset metadata.
  - `_internal.py` — private helpers (`_validate_dates` returns
    `datetime.date`; `_to_offset` normalizes lags to polars offset strings
    like `"1mo"`).
- Missing values in frame outputs are polars nulls, not float `NaN`.
- Do not import pandas inside modules; the only sanctioned touchpoints are
  `backend.py` (boundary conversion) and `regression.py`'s
  `formulaic.model_matrix(model, df.to_pandas())` call.

## Conventions

### Docstrings

- NumPy-style docstrings, parsed by Great Docs (griffe, `parser: numpy`).
- **Examples must use fenced ` ```python ` code blocks, NOT doctest `>>>` /
  `...` prompts.** The Great Docs copy button copies code verbatim, and
  prompts make examples impossible to paste and run. Write:

  ````
  Examples
  --------
  ```python
  import numpy as np
  from tidyfinance import winsorize
  data = np.random.default_rng(123).standard_normal(100)
  winsorized = winsorize(data, 0.05)
  ```
  ````

  Do not reintroduce `>>>` examples. If you ever need verifiable doctests
  with expected output, raise it as a deliberate change — the current
  examples are input-only and carry no doctest assertions.

### Public API

- A function is part of the public API by living in a public (non-`_`)
  submodule as a function/class defined within the package — it is then
  auto-discovered and re-exported from `tidyfinance`.
- To keep something out of the public API and the Reference page, prefix it
  with `_` (module or name).
- Do not name a module `core`, `utils`, `helpers`, `constants`, `config`, or
  `settings` and expect it to appear in the docs: Great Docs auto-excludes
  those names. `core` is kept only because `great-docs.yml` lists it under
  `auto_include`.

### Style

- Ruff, `line-length = 80` (`[tool.ruff]` in `pyproject.toml`). Keep code
  and docstrings within 80 columns.

## Common commands

This project uses `uv`.

```bash
uv run pytest                 # run the test suite (tests/test_*.py)
uv run pytest tests/test_core.py
uv run ruff check .           # lint
uv run ruff format .          # format
```

Every public function should have a matching `tests/test_<name>.py`.

## Documentation (Great Docs)

- Config: `great-docs.yml`. Generated output lives in `great-docs/`
  (git-ignored build artifacts).

```bash
uv run great-docs build       # build the site
uv run great-docs preview --port 3000   # local preview (auto-rebuilds)
uv run great-docs scan        # preview what will be discovered as public API
```

