"""Parse user-supplied stock lists without contacting a provider."""

import csv
from io import StringIO
import re


def parse_stock_codes(text, format="txt"):
    text = text.lstrip("\ufeff")
    if format == "csv":
        rows = list(csv.reader(StringIO(text)))
        rows = [row for row in rows if any(cell.strip() for cell in row)]
        if not rows:
            raise ValueError("Empty stock list")
        headers = [cell.strip().lower() for cell in rows[0]]
        matches = [i for i, name in enumerate(headers) if name in {"code", "代码", "股票代码"}]
        if len(matches) > 1:
            raise ValueError("CSV has ambiguous code columns")
        if matches:
            column, rows = matches[0], rows[1:]
        else:
            if any(len(row) != 1 for row in rows):
                raise ValueError("Multi-column CSV requires a code header")
            column = 0
        tokens = [row[column].strip() if len(row) > column else "" for row in rows]
    elif format == "txt":
        tokens = [s for s in re.split(r"[\s,，;；]+", text.strip()) if s]
    else:
        raise ValueError("Unsupported stock list format")
    codes, invalid, duplicates, normalized = [], [], 0, 0
    seen = set()
    for token in tokens:
        if not re.fullmatch(r"[0-9]{1,6}", token):
            invalid.append(token)
            continue
        code = token.zfill(6)
        normalized += code != token
        if code in seen:
            duplicates += 1
        else:
            seen.add(code)
            codes.append(code)
    if invalid:
        raise ValueError("Invalid stock codes: " + ", ".join(repr(v) for v in invalid[:10]))
    if not 1 <= len(codes) <= 500:
        raise ValueError("Expected 1 to 500 unique stock codes")
    return dict(codes=codes, duplicates=duplicates, normalized=normalized)
