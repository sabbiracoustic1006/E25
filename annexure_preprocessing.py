"""Utilities for converting Annexure-compliant tagged TSV data into aspect rows.

The conversion logic follows the rules highlighted in ``Annexure.pdf``:

* Titles are already tokenised; we must not alter or re-tokenise the ``Token`` column.
* A non-empty ``Tag`` marks the start of a new semantic entity (aspect value).
* An empty ``Tag`` continues the preceding entity and the tokens must be joined
  with a single ASCII space (0x20), regardless of the original whitespace.

These helpers keep the transformation self-contained so the training script can
import a single well-tested implementation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, MutableSequence, Optional, Sequence, Union

try:  # Optional import: allows basic validation without pandas installed.
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - pandas is available during training.
    pd = None  # type: ignore

RequiredColumns = Sequence[str]
_TaggedInput = Union[str, Path, "pd.DataFrame"]

_REQUIRED_COLUMNS: RequiredColumns = (
    "Record Number",
    "Category",
    "Title",
    "Token",
    "Tag",
)


def _load_tagged_dataframe(tagged: _TaggedInput) -> "pd.DataFrame":
    """Return a defensive copy of the tagged data as a pandas DataFrame."""
    if pd is None:  # pragma: no cover - should not trigger in the runtime environment.
        raise ImportError(
            "pandas is required to load the tagged TSV files. "
            "Install pandas or pass a pre-loaded DataFrame instead."
        )

    if isinstance(tagged, (str, Path)):
        source_path = Path(tagged)
        df = pd.read_csv(
            source_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
        )
        # Normalize column headers to avoid BOM/whitespace issues
        df.columns = [str(c).strip() for c in df.columns]
    elif isinstance(tagged, pd.DataFrame):
        df = tagged.copy()
    else:
        raise TypeError(
            "Unsupported input for convert_tagged_to_aspect: "
            "expected a path or pandas.DataFrame."
        )

    missing = [column for column in _REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "Tagged data is missing required columns: {}".format(
                ", ".join(missing)
            )
        )

    return df


def convert_tagged_to_aspect(tagged: _TaggedInput) -> "pd.DataFrame":
    """Collapse token-level annotations into aspect-level rows.

    Parameters
    ----------
    tagged: Union[str, Path, pandas.DataFrame]
        The tagged TSV path or an already-loaded DataFrame that follows the
        competition layout (see Annexure.pdf).

    Returns
    -------
    pandas.DataFrame
        DataFrame with five columns: ``Record Number``, ``Category``, ``Title``,
        ``Aspect Name`` and ``Aspect Value``. Each row corresponds to a single
        semantic entity extracted from the title.
    """
    df = _load_tagged_dataframe(tagged)

    rows: MutableSequence[Mapping[str, str]] = []
    current_record: Optional[str] = None
    current_category: Optional[str] = None
    current_title: Optional[str] = None
    current_tag: Optional[str] = None
    token_buffer: list[str] = []

    def flush_buffer() -> None:
        """Emit the buffered tokens as an aspect row if we have a tag."""
        if current_tag and token_buffer:
            rows.append(
                {
                    "Record Number": current_record,  # type: ignore[arg-type]
                    "Category": current_category,    # type: ignore[arg-type]
                    "Title": current_title,          # type: ignore[arg-type]
                    "Aspect Name": current_tag,
                    "Aspect Value": " ".join(token_buffer),
                }
            )

    # Resolve column indices once to avoid reliance on attribute names
    try:
        idx_record = df.columns.get_loc("Record Number")
        idx_category = df.columns.get_loc("Category")
        idx_title = df.columns.get_loc("Title")
        idx_token = df.columns.get_loc("Token")
        idx_tag = df.columns.get_loc("Tag")
    except KeyError as e:
        raise ValueError(
            "Tagged data missing required column: {}. Found columns: {}".format(
                e, ", ".join(map(str, df.columns))
            )
        )

    for row in df.itertuples(index=False, name=None):
        record_number = row[idx_record]
        category = row[idx_category]
        title = row[idx_title]
        token = row[idx_token]
        tag = row[idx_tag]

        if record_number != current_record:
            flush_buffer()
            current_record = record_number
            current_category = category
            current_title = title
            current_tag = None
            token_buffer = []

        if tag:
            flush_buffer()
            current_tag = tag
            token_buffer = [token]
        else:
            if not current_tag:
                raise ValueError(
                    "Encountered a continuation token without a preceding tag "
                    f"at record {record_number!r}."
                )
            token_buffer.append(token)

    flush_buffer()

    if pd is None:  # pragma: no cover - return type requires pandas.
        raise ImportError(
            "pandas is required to build the aspect DataFrame. Install pandas "
            "before calling convert_tagged_to_aspect."
        )

    return pd.DataFrame(
        rows,
        columns=(
            "Record Number",
            "Category",
            "Title",
            "Aspect Name",
            "Aspect Value",
        ),
    )
