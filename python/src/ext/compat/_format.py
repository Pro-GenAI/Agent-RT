from __future__ import annotations

import string
from collections.abc import Mapping
from typing import Any


class _PlainFormatter(string.Formatter):
    """``str.format`` without attribute/index lookups or positional fields.

    Templates may come from untrusted sources (hubs, config); ``{x.__class__}``
    style lookups would let them walk object graphs and leak secrets.
    """

    def get_field(self, field_name: str, args: Any, kwargs: Any) -> Any:
        if "." in field_name or "[" in field_name:
            raise ValueError(
                f"template field {field_name!r} must be a plain variable name"
            )
        return super().get_field(field_name, args, kwargs)

    def get_value(self, key: Any, args: Any, kwargs: Any) -> Any:
        if isinstance(key, int):
            # A bad template, not a bad argument type: keep ValueError like
            # the attribute/index check above.
            raise ValueError("positional template fields are not supported")  # noqa: TRY004
        return super().get_value(key, args, kwargs)


_FORMATTER = _PlainFormatter()


def safe_format(template: str, values: Mapping[str, Any]) -> str:
    return _FORMATTER.vformat(str(template), (), dict(values))
