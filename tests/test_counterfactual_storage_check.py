"""Pure synthetic storage equivalence; never a market, label or NN test."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from tools import counterfactual_storage_check as candidate


def outcome(function, left, right):
    try:
        return ("returned", function(left, right))
    except Exception as error:
        return (type(error).__name__, str(error))


def pair(case):
    base = np.arange(120, dtype=np.int64)
    if case == "separate":
        return [base], [base.copy()]
    if case == "overlap_views":
        return [base[:70]], [base[50:]]
    if case == "adjacent_views":
        return [base[:10]], [base[10:20]]
    if case == "interleaved":
        return [base[::2]], [base[1::2]]
    if case == "negative_overlap":
        return [base[::-1]], [base[4:10]]
    if case == "negative_separate":
        return [base[::-1]], [base.copy()[::-1]]
    if case == "zero_stride_alias":
        return [np.broadcast_to(base[:1], (20,))], [base[:1].view()]
    if case == "empty_same_object":
        empty = base[:0]
        return [empty], [empty]
    if case == "empty_distinct_views":
        return [base[:0]], [base[:0]]
    if case == "zero_itemsize":
        return [np.empty(4, dtype="V0")], [np.empty(4, dtype="V0")]
    if case == "scalar_overlap":
        return [base[1:2].reshape(())], [base[1:2].reshape(())]
    if case == "readonly_alias":
        left, right = base.view(), base.view()
        left.flags.writeable = right.flags.writeable = False
        return [left], [right]
    if case == "readonly_separate":
        left, right = base.copy(), base.copy()
        left.flags.writeable = right.flags.writeable = False
        return [left], [right]
    if case == "transpose_alias":
        return [base.reshape(12, 10).T], [base[20:30]]
    if case == "structured_holes":
        records = np.zeros(20, dtype=[("a", "i4"), ("b", "i4")])
        return [records["a"]], [records["b"]]
    if case == "partial_byte_overlap":
        bytes_ = np.arange(80, dtype=np.uint8)
        return [bytes_[1:9].view(np.uint16)], [bytes_[8:12]]
    if case == "nested_interval_hidden_alias":
        # A naive two-pointer walk over original arrays skips the short A/B
        # overlap after testing two nonoverlapping strided elements of long A.
        return [base[::10], base[5:6]], [base[5:6].view(), base[85:86]]
    raise AssertionError(case)


@pytest.mark.parametrize("case", ["separate", "overlap_views", "adjacent_views", "interleaved",
    "negative_overlap", "negative_separate", "zero_stride_alias", "empty_same_object",
    "empty_distinct_views", "zero_itemsize", "scalar_overlap", "readonly_alias", "readonly_separate",
    "transpose_alias", "structured_holes", "partial_byte_overlap", "nested_interval_hidden_alias"])
def test_exact_same_return_or_exception_as_frozen_checker(case):
    left, right = pair(case)
    assert outcome(candidate.assert_independent_storage, left, right) == outcome(candidate.original._independent, left, right)


def test_disjoint_unions_fast_accept_and_strided_false_overlap_falls_back():
    left, right = pair("separate")
    assert candidate.array_bounds_prove_disjoint(left, right) is True
    assert candidate.assert_independent_storage(left, right) is None
    left, right = pair("interleaved")
    assert candidate.array_bounds_prove_disjoint(left, right) is False
    assert candidate.assert_independent_storage(left, right) is None


@pytest.mark.parametrize("kind", ["dict", "list", "set", "empty_array", "generator", "random_state"])
def test_shared_mutable_objects_are_rejected_even_with_no_nonempty_array_storage(kind):
    shared = {"dict": {}, "list": [], "set": set(), "empty_array": np.empty(0),
              "generator": np.random.default_rng(42), "random_state": np.random.RandomState(42)}[kind]
    left, right = {"mutable": shared}, {"mutable": shared}
    assert outcome(candidate.assert_independent_storage, left, right) == ("RuntimeError", "Execution books share mutable objects")
    assert outcome(candidate.assert_independent_storage, left, right) == outcome(candidate.original._independent, left, right)


def test_object_arrays_with_disjoint_pointer_storage_still_reject_shared_nested_cells():
    shared = {"items": []}
    left, right = np.empty(1, dtype=object), np.empty(1, dtype=object)
    left[0] = right[0] = shared
    assert candidate.array_bounds_prove_disjoint([left], [right]) is True
    assert outcome(candidate.assert_independent_storage, left, right) == ("RuntimeError", "Execution books share mutable objects")


def test_deepcopy_object_array_really_copies_nested_mutables():
    left = np.empty(1, dtype=object)
    left[0] = {"items": [1, 2]}
    right = deepcopy(left)
    assert candidate.assert_independent_storage(left, right) is None
    assert right[0] is not left[0] and right[0]["items"] is not left[0]["items"]


def test_pandas_deepcopy_nested_cells_is_not_mistaken_for_independent_storage():
    frame = pd.DataFrame({"cell": [{"items": [1, 2]}]})
    copied = deepcopy(frame)
    assert frame.iat[0, 0] is copied.iat[0, 0]
    assert outcome(candidate.assert_independent_storage, frame, copied) == outcome(candidate.original._independent, frame, copied)
    assert outcome(candidate.assert_independent_storage, frame, copied)[0] == "RuntimeError"


def test_dataframe_attrs_shared_mutable_is_rejected_even_when_blocks_are_independent():
    shared = []
    left, right = pd.DataFrame({"x": [1., 2.]}), pd.DataFrame({"x": [1., 2.]})
    left.attrs["shared"] = right.attrs["shared"] = shared
    assert outcome(candidate.assert_independent_storage, left, right) == outcome(candidate.original._independent, left, right)
    assert outcome(candidate.assert_independent_storage, left, right)[0] == "RuntimeError"


def test_numeric_dataframe_deepcopy_returns_exactly_like_original():
    left = pd.DataFrame(np.arange(40, dtype=np.float64).reshape(10, 4))
    right = deepcopy(left)
    assert candidate.assert_independent_storage(left, right) is None
    assert candidate.original._independent(left, right) is None


def test_array_subclass_dispatch_is_not_bypassed():
    class CustomShareError(RuntimeError):
        pass

    class Array(np.ndarray):
        def __array_function__(self, function, types, args, kwargs):
            if function is np.shares_memory:
                raise CustomShareError("synthetic subclass exact-dispatch sentinel")
            return super().__array_function__(function, types, args, kwargs)

    left = np.arange(4).view(Array)
    right = np.arange(4).view(Array)
    assert candidate.array_bounds_prove_disjoint([left], [right]) is False
    assert outcome(candidate.assert_independent_storage, left, right) == outcome(candidate.original._independent, left, right)
    assert outcome(candidate.assert_independent_storage, left, right)[0] == "CustomShareError"


def test_many_independent_arrays_do_not_mutate_during_either_check():
    left = [np.arange(32, dtype=np.float64) + number for number in range(400)]
    right = [value.copy() for value in left]
    before = [value.tobytes() for value in left + right]
    assert candidate.array_bounds_prove_disjoint(left, right) is True
    assert candidate.assert_independent_storage(left, right) is None
    assert candidate.original._independent(left, right) is None
    assert before == [value.tobytes() for value in left + right]
@pytest.mark.parametrize("shared_cell", [False, True])
def test_structured_object_fields_are_explicitly_unsupported(shared_cell):
    left = np.empty(1, dtype=[("cell", object), ("number", "i8")])
    right = np.empty_like(left)
    left["cell"][0] = {"items": [1]}
    right["cell"][0] = left["cell"][0] if shared_cell else {"items": [1]}
    left["number"] = right["number"] = 7
    with pytest.raises(TypeError, match="Structured arrays with object fields"):
        candidate.assert_independent_storage(left, right)

