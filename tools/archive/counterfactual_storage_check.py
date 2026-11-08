"""Unused performance candidate; the running label pilot does not import this.

Only a new registered batch may adopt this storage checker. The frozen helper
supplies its existing mutable-object traversal, including plain object cells.
Structured arrays containing object fields are unsupported and rejected here.
"""

import numpy as np

from tools.archive import timing_counterfactual as original

STORAGE_CHECK_VERSION = "byte_union_disjoint_else_original_exact_v1"


def _byte_bounds(array):
    function = getattr(np, "byte_bounds", None)
    if function is None:
        function = np.lib.array_utils.byte_bounds
    return function(array)


def _merged_bounds(arrays):
    intervals = []
    for array in arrays:
        # Preserve custom ndarray subclass dispatch and exception semantics by
        # leaving those arrays to the original exact check.
        if type(array) is not np.ndarray:
            return None
        if array.size == 0 or array.dtype.itemsize == 0:
            continue
        lower, upper = _byte_bounds(array)
        lower, upper = int(lower), int(upper)
        if upper < lower:
            return None
        if upper > lower:
            intervals.append((lower, upper))
    intervals.sort()
    merged = []
    for lower, upper in intervals:
        if merged and lower <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], upper))
        else:
            merged.append((lower, upper))
    return merged


def array_bounds_prove_disjoint(left_arrays, right_arrays):
    """True proves disjoint byte storage; False requires the old exact check."""
    left, right = _merged_bounds(left_arrays), _merged_bounds(right_arrays)
    if left is None or right is None:
        return False
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i][1] <= right[j][0]:
            i += 1
        elif right[j][1] <= left[i][0]:
            j += 1
        else:
            return False
    return True


def assert_independent_storage(left, right):
    """Preserve frozen isolation checks within the supported state types.

    Interval unions only avoid exact calls when no byte ranges can intersect.
    Overlapping bounds, including harmless strided holes, use the original full
    pairwise `shares_memory` predicate and retain its exceptions unchanged.
    """
    left_ids, left_arrays = original._objects(left)
    right_ids, right_arrays = original._objects(right)
    for array in left_arrays + right_arrays:
        if array.dtype.hasobject and array.dtype.fields is not None:
            raise TypeError("Structured arrays with object fields are unsupported")
    if left_ids.intersection(right_ids):
        raise RuntimeError("Execution books share mutable objects")
    if array_bounds_prove_disjoint(left_arrays, right_arrays):
        return None
    if any(np.shares_memory(old, new) for old in left_arrays for new in right_arrays):
        raise RuntimeError("Execution books share mutable array storage")
    return None
