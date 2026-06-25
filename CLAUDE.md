# CLAUDE.md

Guidance for working in this repository (`tidyfinance`, the Python companion
package to the *Tidy Finance with Python* book).

## Project layout

- `tidyfinance/` — the package.
  - `__init__.py` — builds the public API automatically by scanning the
    public submodules and re-exporting their functions/classes (see
    `__all__`). Data-bearing functions are wrapped at this boundary by
    `backend._use_backend` so they honor the active polars/pandas backend.
    Keep this file's namespace clean: discovery loop variables, imports
    (`importlib`, `pkgutil`, `types`), and internal toggles are deleted or
    underscore-prefixed at the end so they don't leak into `dir(tidyfinance)`
    or the docs.
  - `core.py` — analytics functions (portfolio sorts, breakpoints, beta /
    Fama-MacBeth estimation, lagging, summary statistics).
  - `data_download.py` — `download_data` and the WRDS / Fama-French / FRED /
    OSAP / Hugging Face download helpers.
  - `backend.py` — `set_backend` / `get_backend` and the internal
    `_use_backend` decorator (private — must stay underscore-prefixed so it
    is not re-exported or documented).
  - `utilities.py`, `supported_datasets.py` — helpers and dataset metadata.
  - `_internal.py`, `_pseudo.py` — private modules (leading underscore =
    not scanned for the public API).

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

- The navbar version badge comes from the latest **GitHub Release**, not
  `pyproject.toml` — there is no option to read it from `pyproject.toml`.
  Publish a release to update it.
