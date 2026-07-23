from model_runtime import get_runtime


def test_cache_resource_reuses_exact_model():
    get_runtime.clear()
    first = get_runtime()
    second = get_runtime()
    assert first is second
    assert first.model is second.model


def test_required_genuine_model_generations():
    runtime = get_runtime()
    cases = [
        (347, 928, "1275", "5721"),
        (499, 501, "1000", "0001"),
        (91, 909, "1000", "0001"),
        (999, 999, "1998", "8991"),
    ]
    for a, b, answer, internal in cases:
        details = runtime.add_with_details(a, b)
        assert details["answer"] == answer
        assert details["internal_digits"] == internal
