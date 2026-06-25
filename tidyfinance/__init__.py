"""tidyfinance public API.

Symbols are discovered automatically from this package's submodules.
Two toggles control what is exposed:

- _INCLUDE_PRIVATE_MODULES: scan submodules whose name starts with '_'
  (e.g. '_internal', '_pseudo') in addition to public ones.
- _INCLUDE_PRIVATE_NAMES: within each scanned module, expose attributes
  whose name starts with '_' (e.g. '_download_data_wrds_crsp') in
  addition to the public ones.

Set both to False for a strictly public API. Set both to True to mirror
the previous behaviour where '_download_data_*' helpers and internal
utilities were importable directly from 'tidyfinance'.
"""

import importlib
import pkgutil
import types

# Toggle these to control auto-discovery scope.
_INCLUDE_PRIVATE_MODULES = False  # scan '_internal', '_pseudo', etc.
_INCLUDE_PRIVATE_NAMES = False  # expose '_download_data_*' etc.

# Names to never expose, even if the filters above would include them.
# Use this for things like '__future__' re-imports or sentinel objects.
_EXCLUDE = {
    "annotations",
}

__all__ = []
_seen = set()

if "__path__" in globals():
    for _finder, _module_name, _ispkg in pkgutil.iter_modules(__path__):
        if _module_name.startswith("_") and not _INCLUDE_PRIVATE_MODULES:
            continue

        _module = importlib.import_module(f".{_module_name}", package=__name__)

        for _name in dir(_module):
            if _name.startswith("__") and _name.endswith("__"):
                # Skip dunders unconditionally.
                continue
            if _name.startswith("_") and not _INCLUDE_PRIVATE_NAMES:
                continue
            if _name in _EXCLUDE or _name in _seen:
                continue

            _obj = getattr(_module, _name)
            # Re-export only functions and classes. Skip module
            # re-exports (e.g. 'import os') and constants.
            if not isinstance(_obj, (types.FunctionType, type)):
                continue
            # Skip objects imported from third-party packages (e.g.
            # 'create_engine', 'ThreadPoolExecutor'); only expose
            # symbols defined within this package.
            if not getattr(_obj, "__module__", "").startswith(__name__):
                continue

            globals()[_name] = _obj
            __all__.append(_name)
            _seen.add(_name)

# Wrap the public, data-bearing API at the boundary so it honors the
# active backend (see tidyfinance.set_backend). The wrapped functions
# accept polars or pandas input and return the active backend's data
# frame type. Internal calls between core functions are left untouched,
# since they reference the original (undecorated) functions in their
# defining modules.
from .backend import _use_backend  # noqa: E402

_BACKEND_WRAPPED = (
    "add_lagged_columns",
    "assign_portfolio",
    "compute_breakpoints",
    "compute_portfolio_returns",
    "compute_long_short_returns",
    "compute_rolling_value",
    "create_summary_statistics",
    "estimate_betas",
    "estimate_fama_macbeth",
    "estimate_model",
    "join_lagged_values",
    "filter_sorting_data",
    "implement_portfolio_sort",
    "download_data",
    "list_supported_datasets",
    "list_supported_indexes",
    "process_trace_data",
)
for _name in _BACKEND_WRAPPED:
    if _name not in globals():
        raise RuntimeError(
            f"'{_name}' is listed in _BACKEND_WRAPPED but was not "
            f"auto-discovered from any public submodule. Check the "
            f"name for typos or update _BACKEND_WRAPPED."
        )
    globals()[_name] = _use_backend(globals()[_name])

# Clean up the module namespace so only the public API (the names in
# __all__) remains as a package attribute. This keeps dir(tidyfinance)
# and documentation tooling free of import helpers, discovery
# internals, and leftover loop variables. Submodules stay importable
# via 'import tidyfinance.<submodule>'.
del importlib, pkgutil, types, _use_backend
del _seen
for _leaked in (
    "_finder",
    "_ispkg",
    "_module_name",
    "_module",
    "_name",
    "_obj",
    "_leaked",
):
    globals().pop(_leaked, None)
