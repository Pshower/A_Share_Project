"""Explain SharedStockNetwork using small, deterministic synthetic observations.

Run from the repository root:
    python -m tests.test_shared_stock_network_walkthrough
    python -m pytest tests/test_shared_stock_network_walkthrough.py -q

Hooks capture the real forward pass. No model training or market files are used.
"""

import torch

from src.agents.ppo_policy import SharedStockNetwork


STOCKS = ("A", "B", "C", "D")
FEATURES = ("return", "momentum_20", "volatility_20")
HIDDEN_SIZES = (8, 4)


def make_observation():
    return {
        "stock_features": torch.tensor([[
            [[0.10, -0.20, 0.30], [0.20, 0.20, 0.40]],
            [[0.30, 0.70, 0.20], [0.40, 0.90, 0.30]],
            [[0.20, 0.50, 0.40], [0.30, 0.80, 0.20]],
            [[0.00, 0.00, 0.00], [0.00, 0.00, 0.00]],
        ]], dtype=torch.float32),
        "portfolio": torch.tensor([[0.20, 0.00, 0.30, 0.10, 0.35]]),
        "valid_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.int8),
        "buy_mask": torch.tensor([[1, 0, 1, 0]], dtype=torch.int8),
        "sell_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.int8),
        "receivable_ratio": torch.tensor([[0.05]]),
    }


def make_network(with_prior=False):
    # Avoid changing the random seed of unrelated tests.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        network = SharedStockNetwork(
            lookback=2,
            n_features=len(FEATURES),
            hidden_sizes=HIDDEN_SIZES,
            initial_log_std=-3.0,
            prior_feature_index=FEATURES.index("momentum_20") if with_prior else None,
            prior_strength=4.0,
            residual_scale=0.25,
        )
    return network.eval()


def capture_forward(network, observation):
    trace, hooks = {}, []
    for name in ("encoder", "stock_head", "cash_head", "value_head"):
        module = getattr(network, name)

        def before(_module, inputs, name=name):
            trace[name + "_input"] = inputs[0].detach().clone()

        def after(_module, _inputs, output, name=name):
            trace[name + "_output"] = output.detach().clone()

        hooks.append(module.register_forward_pre_hook(before))
        hooks.append(module.register_forward_hook(after))
    try:
        with torch.inference_mode():
            outputs = network(observation)
    finally:
        for hook in hooks:
            hook.remove()
    return outputs, trace


def test_features_are_flattened_per_stock_and_flags_keep_their_order():
    network = make_network()
    outputs, trace = capture_forward(network, make_observation())
    torch.testing.assert_close(trace["encoder_input"][0, 0], torch.tensor([
        0.10, -0.20, 0.30, 0.20, 0.20, 0.40, 0.20, 1.0, 1.0, 1.0,
    ]))
    assert trace["encoder_input"].shape == (1, 4, 10)
    assert trace["encoder_output"].shape == (1, 4, 4)
    assert trace["stock_head_input"].shape == (1, 4, 14)
    assert trace["cash_head_input"].shape == (1, 10)
    assert [tuple(value.shape) for value in outputs] == [(1, 5), (1, 1), (5,)]
    assert all(torch.isfinite(value).all() for value in outputs)
    assert all(not value.requires_grad for value in outputs)


def test_context_excludes_invalid_market_data_but_preserves_held_assets():
    obs = make_observation()
    _, trace = capture_forward(make_network(), obs)
    encoded, context = trace["encoder_output"], trace["cash_head_input"]
    torch.testing.assert_close(context[:, :4], encoded[:, :3].mean(dim=1))
    expected_held = (encoded * obs["portfolio"][:, :-1].unsqueeze(-1)).sum(dim=1)
    torch.testing.assert_close(context[:, 4:8], expected_held)
    held_without_d = (encoded[:, :3] * obs["portfolio"][:, :3].unsqueeze(-1)).sum(dim=1)
    assert not torch.allclose(expected_held, held_without_d)
    torch.testing.assert_close(context[:, 8:], torch.tensor([[0.35, 0.05]]))
    torch.testing.assert_close(trace["value_head_input"], context)
    torch.testing.assert_close(trace["stock_head_input"][:, :, 4:], context.unsqueeze(1).expand(-1, 4, -1))


def test_portfolio_changes_reach_encoder_context_and_predictions():
    network, obs = make_network(), make_observation()
    before, original = capture_forward(network, obs)
    moved = {key: value.clone() for key, value in obs.items()}
    moved["portfolio"][0, :3] = torch.tensor([0.35, 0.00, 0.15])
    after, changed = capture_forward(network, moved)
    torch.testing.assert_close(changed["encoder_input"][:, :, :6], original["encoder_input"][:, :, :6])
    assert not torch.allclose(changed["encoder_output"], original["encoder_output"])
    assert not torch.allclose(changed["cash_head_input"], original["cash_head_input"])
    assert not torch.allclose(after[0], before[0])


def test_stock_order_permutations_preserve_corresponding_predictions():
    network, obs = make_network(), make_observation()
    order = [2, 0, 3, 1]
    moved = {key: value.clone() for key, value in obs.items()}
    for key in ("stock_features", "valid_mask", "buy_mask", "sell_mask"):
        moved[key] = obs[key][:, order]
    moved["portfolio"] = torch.cat([obs["portfolio"][:, order], obs["portfolio"][:, -1:]], dim=1)
    original, _ = capture_forward(network, obs)
    changed, _ = capture_forward(network, moved)
    torch.testing.assert_close(changed[0][:, :-1], original[0][:, order])
    torch.testing.assert_close(changed[0][:, -1:], original[0][:, -1:])
    torch.testing.assert_close(changed[1], original[1])
    torch.testing.assert_close(changed[2], original[2])


def test_prior_ranks_only_eligible_stocks_using_the_latest_time_step():
    network, obs = make_network(with_prior=True), make_observation()
    outputs, trace = capture_forward(network, obs)
    assert torch.count_nonzero(trace["stock_head_output"]) == 0
    assert torch.count_nonzero(trace["cash_head_output"]) == 0
    # A and C are eligible; B's higher momentum is excluded by buy_mask.
    torch.testing.assert_close(outputs[0], torch.tensor([[-4.0, 0.0, 4.0, 0.0, 0.0]]))
    torch.testing.assert_close(outputs[2], torch.full((5,), -3.0))
    changed = {key: value.clone() for key, value in obs.items()}
    changed["stock_features"][:, :, 0, 1] = torch.tensor([[100.0, 200.0, -100.0, 0.0]])
    # Older features enter the encoder, but the initialized prior uses the latest signal only.
    altered, _ = capture_forward(network, changed)
    torch.testing.assert_close(altered[0], outputs[0])


def test_no_valid_stocks_produces_finite_zero_market_summary():
    obs = make_observation()
    obs["stock_features"].zero_()
    obs["valid_mask"].zero_()
    obs["buy_mask"].zero_()
    outputs, trace = capture_forward(make_network(with_prior=True), obs)
    torch.testing.assert_close(trace["cash_head_input"][:, :4], torch.zeros((1, 4)))
    torch.testing.assert_close(outputs[0], torch.zeros((1, 5)))
    assert all(torch.isfinite(value).all() for value in outputs)


def show(label, tensor):
    value = tensor[0] if tensor.ndim > 1 else tensor
    print(f"{label}: shape={tuple(tensor.shape)}")
    print(value.detach().cpu().numpy().round(4))


def main():
    obs, network = make_observation(), make_network()
    outputs, trace = capture_forward(network, obs)
    print("SharedStockNetwork walkthrough: synthetic data, fixed seed, CPU forward only")
    print("B=1, N=4, L=2, F=3, encoder hidden sizes=(8, 4)")
    print(f"Stocks={STOCKS}; feature order={FEATURES}")
    print("A/C: valid and buyable; B: valid but not buyable; D: invalid with 10% holdings.")
    print("Stock weights=60%, cash=35%, receivables=5%.")

    print("\n1. Input [batch, stock, day, feature]")
    show("stock_features", obs["stock_features"])
    print("\n2. Flatten only day/feature axes, append [weight, valid, buy, sell]")
    show("encoder_input", trace["encoder_input"])
    print("\n3. Apply the SAME encoder to every stock: 10 -> 8 -> 4")
    show("encoded", trace["encoder_output"])

    print("\n4. Pool encoded stocks into portfolio context")
    context = trace["cash_head_input"]
    show("market: mean of valid A/B/C encodings", context[:, :4])
    show("held: weighted sum INCLUDING held invalid D", context[:, 4:8])
    show("context: [market(4), held(4), cash(1), receivable(1)]", context)
    print("The encoder includes holdings flags, so even the market summary can change with holdings.")

    print("\n5. Each stock receives its own encoding plus the SAME context")
    show("stock_head_input: 14 -> 4 -> 1", trace["stock_head_input"])
    show("mean: [A, B, C, D, cash]", outputs[0])
    show("value: discounted future reward estimate", outputs[1])
    show("log_std: shared stock scale plus cash scale", outputs[2])
    show("std = exp(log_std)", outputs[2].exp())

    print("\n6. Move 15% holdings from C to A, keeping market features unchanged")
    moved = {key: value.clone() for key, value in obs.items()}
    moved["portfolio"][0, :3] = torch.tensor([0.35, 0.00, 0.15])
    changed, changed_trace = capture_forward(network, moved)
    show("original portfolio", obs["portfolio"])
    show("changed portfolio", moved["portfolio"])
    show("changed context", changed_trace["cash_head_input"])
    show("changed mean", changed[0])

    print("\n7. Enable momentum prior, with actor output layers initialized to zero")
    prior, prior_trace = capture_forward(make_network(with_prior=True), obs)
    show("latest momentum_20", obs["stock_features"][:, :, -1, 1])
    show("raw neural stock head", prior_trace["stock_head_output"])
    show("prior mean: only A/C ranked, low=-4 and high=+4", prior[0])
    print("These means parameterize latent actions; map_action computes target weights later.")

    print("\n8. Verify processing and invariance properties")
    checks = (
        test_features_are_flattened_per_stock_and_flags_keep_their_order,
        test_context_excludes_invalid_market_data_but_preserves_held_assets,
        test_portfolio_changes_reach_encoder_context_and_predictions,
        test_stock_order_permutations_preserve_corresponding_predictions,
        test_prior_ranks_only_eligible_stocks_using_the_latest_time_step,
        test_no_valid_stocks_produces_finite_zero_market_summary,
    )
    for check in checks:
        check()
        print("PASS " + check.__name__)
    print(f"Completed: {len(checks)} checks passed.")


if __name__ == "__main__":
    main()
