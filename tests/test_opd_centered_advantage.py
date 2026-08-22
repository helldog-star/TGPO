"""Unit tests for OPD² Eq.(6) centered OPD advantage.

Run:  python -m pytest tests/test_opd_centered_advantage.py -q
  or: python tests/test_opd_centered_advantage.py
"""
import os
import sys
import types
import importlib.util

import torch


def _masked_mean(values, mask, axis=None):
    return (values * mask).sum(axis) / (mask.sum(axis) + 1e-8)


def _load_mod():
    for name in ["verl", "verl.utils", "verl.utils.torch_functional"]:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["verl.utils.torch_functional"].masked_mean = _masked_mean
    path = os.path.join(
        os.path.dirname(__file__),
        "..", "luffy", "verl", "verl", "mix_src", "mix_core_alg.py",
    )
    spec = importlib.util.spec_from_file_location("mix_core_alg", os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load_mod()


def _paper_example():
    # tokens {6,7,8,9} with student / teacher probs from the Eq.(6) walkthrough
    stu_p = torch.tensor([0.4, 0.3, 0.2, 0.1])
    tea_p = torch.tensor([0.7, 0.1, 0.1, 0.1])
    return stu_p, tea_p, stu_p.log(), tea_p.log()


def test_paper_example_sampled_token_6():
    stu_p, tea_p, stu_lp, tea_lp = _paper_example()
    reward = tea_lp - stu_lp
    baseline = (stu_p * reward).sum()
    # sampled token = "6" (index 0)
    adv, _, metrics = MOD.compute_opd_centered_advantage(
        old_log_probs=stu_lp[0:1].view(1, 1),
        teacher_log_prob=tea_lp[0:1].view(1, 1),
        response_mask=torch.ones(1, 1),
        student_topk_log_probs=stu_lp.view(1, 1, 4),
        teacher_topk_log_probs=tea_lp.view(1, 1, 4),
    )
    expected = reward[0] - baseline
    assert torch.allclose(adv.squeeze(), expected, atol=1e-5)
    assert expected.item() > 0                       # "6" is above the student mean
    assert abs(metrics["opd/topk_mass_mean"] - 1.0) < 1e-5


def test_paper_example_sampled_token_7():
    stu_p, tea_p, stu_lp, tea_lp = _paper_example()
    reward = tea_lp - stu_lp
    baseline = (stu_p * reward).sum()
    adv, _, _ = MOD.compute_opd_centered_advantage(
        old_log_probs=stu_lp[1:2].view(1, 1),
        teacher_log_prob=tea_lp[1:2].view(1, 1),
        response_mask=torch.ones(1, 1),
        student_topk_log_probs=stu_lp.view(1, 1, 4),
        teacher_topk_log_probs=tea_lp.view(1, 1, 4),
    )
    expected = reward[1] - baseline
    assert torch.allclose(adv.squeeze(), expected, atol=1e-5)
    assert expected.item() < 0                       # "7" is below the student mean


def test_uncentered_rkl_is_raw_opd_reward():
    stu_p, tea_p, stu_lp, tea_lp = _paper_example()
    mask = torch.ones(1, 1)
    raw, _ = MOD.compute_rkl_advantage(
        old_log_probs=stu_lp[0:1].view(1, 1),
        teacher_log_prob=tea_lp[0:1].view(1, 1),
        response_mask=mask,
    )
    assert torch.allclose(raw.squeeze(), tea_lp[0] - stu_lp[0], atol=1e-6)


def test_full_vocab_policy_mean_advantage_is_zero():
    """Σ_v π_θ(v) A(v) = 0 when the top-k support is the full vocab."""
    stu_p, tea_p, stu_lp, tea_lp = _paper_example()
    k = stu_p.numel()
    advs = []
    for i in range(k):
        adv, _, _ = MOD.compute_opd_centered_advantage(
            old_log_probs=stu_lp[i:i + 1].view(1, 1),
            teacher_log_prob=tea_lp[i:i + 1].view(1, 1),
            response_mask=torch.ones(1, 1),
            student_topk_log_probs=stu_lp.view(1, 1, k),
            teacher_topk_log_probs=tea_lp.view(1, 1, k),
        )
        advs.append(adv.squeeze())
    mean_adv = (stu_p * torch.stack(advs)).sum()
    assert torch.allclose(mean_adv, torch.zeros(()), atol=1e-5)


def test_constant_reward_shift_does_not_change_advantage():
    """On a complete support, R → R+c leaves A unchanged."""
    stu_p, tea_p, stu_lp, tea_lp = _paper_example()
    c = 2.5
    kwargs = dict(
        old_log_probs=stu_lp[0:1].view(1, 1),
        teacher_log_prob=tea_lp[0:1].view(1, 1),
        response_mask=torch.ones(1, 1),
        student_topk_log_probs=stu_lp.view(1, 1, 4),
        teacher_topk_log_probs=tea_lp.view(1, 1, 4),
    )
    a0, _, _ = MOD.compute_opd_centered_advantage(**kwargs)
    kwargs_c = dict(kwargs)
    kwargs_c["teacher_log_prob"] = kwargs["teacher_log_prob"] + c
    kwargs_c["teacher_topk_log_probs"] = kwargs["teacher_topk_log_probs"] + c
    a1, _, _ = MOD.compute_opd_centered_advantage(**kwargs_c)
    assert torch.allclose(a0, a1, atol=1e-5)


def test_response_mask_zeros_padded_tokens():
    bs, l, k = 2, 3, 4
    old_lp = torch.randn(bs, l)
    tea_lp = torch.randn(bs, l)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    adv, _, _ = MOD.compute_opd_centered_advantage(
        old_log_probs=old_lp,
        teacher_log_prob=tea_lp,
        response_mask=mask,
        student_topk_log_probs=torch.randn(bs, l, k),
        teacher_topk_log_probs=torch.randn(bs, l, k),
    )
    assert torch.equal(adv[:, 2], torch.zeros(bs))
    assert adv[1, 1].item() == 0.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
