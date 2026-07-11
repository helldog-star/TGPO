"""Regression tests for the TGPO / reverse-KL teacher-regularization loss paths.

Exercises the REAL `compute_token_on_tgpo_loss` from mix_core_alg.py on synthetic
CPU tensors (no cluster / models / data needed). Only `verl_F.masked_mean` is stubbed.

Covers, for each teacher-reg variant:
  - forward CE (hard label, plain TGPO)            kl_direction="forward", no top-k
  - forward top-k KL (soft distribution)           kl_direction="forward" + top-k
  - reverse top-k KL (distribution level)          kl_direction="reverse"
  - reverse K1 (score-function; ex use_rkl_reg_loss) kl_direction="reverse_k1"
  - reverse K2 / K3 (pointwise)                    kl_direction="reverse_k2" / "reverse_k3"

Checks: runs & finite; reg_only drops pg_loss (total==coef*reg); +GRPO gives total==pg+coef*reg;
gradients are pathwise (into student / log_prob) with the teacher detached; and the divergence
actually decreases under SGD (a valid minimization target).

Run:  python -m pytest tests/test_tgpo_losses.py -q
  or: python tests/test_tgpo_losses.py
"""
import os
import sys
import types
import importlib.util

import torch
import torch.nn.functional as Fnn


# --------------------------------------------------------------------------- #
# Load the real loss module, stubbing the single verl helper it references.
# --------------------------------------------------------------------------- #
def _masked_mean(values, mask, axis=None):
    return (values * mask).sum(axis) / (mask.sum(axis) + 1e-8)


def _load_loss_fn():
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
    return mod.compute_token_on_tgpo_loss


LOSS_FN = _load_loss_fn()

BS, L, K, V = 2, 4, 5, 20
COEF = 0.1


def _common():
    torch.manual_seed(0)
    return {
        "eos": torch.ones(BS, L),
        "old_lp": torch.randn(BS, L),
        "adv": torch.randn(BS, L),
        "t_ids_lp": torch.randn(BS, L),
    }


def _make_topk(student_logits, teacher_logits):
    """teacher_topk_logits (raw, loss re-softmaxes) + student logp @ teacher top-k ids."""
    t_topk_lp, t_topk_ids = torch.topk(Fnn.log_softmax(teacher_logits, dim=-1), K, dim=-1)
    teacher_topk_logits = torch.gather(teacher_logits, -1, t_topk_ids)
    stu_topk_lp = torch.gather(Fnn.log_softmax(student_logits, dim=-1), -1, t_topk_ids)
    return teacher_topk_logits, stu_topk_lp


# --------------------------------------------------------------------------- #
# forward CE (no top-k) == plain TGPO hard-label
# --------------------------------------------------------------------------- #
def test_forward_ce_fallback():
    c = _common()
    teacher_ids_lp = torch.randn(BS, L, requires_grad=True)
    total, pg, reg, _, _ = LOSS_FN(
        old_log_prob=c["old_lp"], log_prob=c["old_lp"].clone().requires_grad_(True),
        advantages=c["adv"], eos_mask=c["eos"], teacher_ids_log_probs=teacher_ids_lp,
        cliprange=0.2, teacher_coef=COEF, kl_direction="forward", reg_only=True,
    )
    total.backward()
    assert torch.isfinite(total)
    # reg is exactly the hard-label CE
    assert torch.allclose(reg, -_masked_mean(teacher_ids_lp, c["eos"]))
    assert torch.allclose(total, COEF * reg)                       # reg_only drops pg_loss
    assert teacher_ids_lp.grad is not None                          # pathwise into student logp


# --------------------------------------------------------------------------- #
# forward top-k KL (soft distribution) + GRPO
# --------------------------------------------------------------------------- #
def test_forward_topk_mixed_with_grpo():
    c = _common()
    log_prob = c["old_lp"].clone().requires_grad_(True)
    stu_logits = torch.randn(BS, L, V, requires_grad=True)
    tea_topk_logits, stu_topk_lp = _make_topk(stu_logits, torch.randn(BS, L, V))
    total, pg, reg, _, _ = LOSS_FN(
        old_log_prob=c["old_lp"], log_prob=log_prob, advantages=c["adv"], eos_mask=c["eos"],
        teacher_ids_log_probs=c["t_ids_lp"], cliprange=0.2, teacher_coef=COEF,
        teacher_topk_logits=tea_topk_logits, student_teacher_topk_log_probs=stu_topk_lp,
        kl_direction="forward", reg_only=False,
    )
    total.backward()
    assert torch.isfinite(total)
    assert reg.item() >= -1e-6                                     # strict top-k forward KL >= 0
    assert torch.allclose(total, pg + COEF * reg)                  # mixed with GRPO
    assert stu_logits.grad is not None and torch.isfinite(stu_logits.grad).all()   # (A) into student
    assert log_prob.grad is not None and torch.isfinite(log_prob.grad).all()       # pg_loss into log_prob
    assert not tea_topk_logits.requires_grad                        # teacher detached


def test_forward_topk_sgd_decreases():
    c = _common()
    stu_logits = torch.randn(BS, L, V, requires_grad=True)
    tea_logits = torch.randn(BS, L, V)
    opt = torch.optim.SGD([stu_logits], lr=1.0)
    first = last = None
    for step in range(200):
        opt.zero_grad()
        tk_logits, s_topk_lp = _make_topk(stu_logits, tea_logits)
        _, _, reg, _, _ = LOSS_FN(
            old_log_prob=c["old_lp"], log_prob=c["old_lp"], advantages=c["adv"], eos_mask=c["eos"],
            teacher_ids_log_probs=c["t_ids_lp"], cliprange=0.2, teacher_coef=1.0,
            teacher_topk_logits=tk_logits, student_teacher_topk_log_probs=s_topk_lp,
            kl_direction="forward", reg_only=True,
        )
        reg.backward(); opt.step()
        first = reg.item() if step == 0 else first
        last = reg.item()
    assert last < first - 0.5                                       # clearly decreasing


# --------------------------------------------------------------------------- #
# reverse top-k KL (distribution level, renormalized on support -> proper KL >= 0)
# --------------------------------------------------------------------------- #
def test_reverse_topk_reg_only():
    c = _common()
    log_prob = c["old_lp"].clone().requires_grad_(True)
    teacher_topk_logits = torch.randn(BS, L, K)
    stu_topk_lp = torch.randn(BS, L, K, requires_grad=True)
    total, pg, reg, _, _ = LOSS_FN(
        old_log_prob=c["old_lp"], log_prob=log_prob, advantages=c["adv"], eos_mask=c["eos"],
        teacher_ids_log_probs=c["t_ids_lp"], cliprange=0.2, teacher_coef=COEF,
        teacher_topk_logits=teacher_topk_logits, student_teacher_topk_log_probs=stu_topk_lp,
        kl_direction="reverse", reg_only=True,
    )
    total.backward()
    assert torch.isfinite(total)
    assert reg.item() >= -1e-6                                      # renormalized -> proper KL >= 0
    assert torch.allclose(total, COEF * reg)                       # reg_only drops pg_loss
    assert stu_topk_lp.grad is not None and torch.isfinite(stu_topk_lp.grad).all()
    assert not teacher_topk_logits.requires_grad


def test_reverse_topk_teacher_score_shift_invariant():
    """Reverse branch renormalizes teacher scores over the support via log_softmax, so it is
    invariant to a per-position additive constant. This is what lets the teacher worker feed
    teacher *log-probs* (logits - logZ_full) at the student top-k into the `teacher_topk_logits`
    slot and get the correct KL(student-top-k support) -- the basis of reverse-on-student-topk."""
    c = _common()
    logits = torch.randn(BS, L, K)
    shift = torch.randn(BS, L, 1)                      # per-position constant (e.g. -logZ_full)
    stu = torch.randn(BS, L, K)
    kw = dict(old_log_prob=c["old_lp"], log_prob=c["old_lp"], advantages=c["adv"], eos_mask=c["eos"],
              teacher_ids_log_probs=c["t_ids_lp"], cliprange=0.2, teacher_coef=1.0,
              student_teacher_topk_log_probs=stu, kl_direction="reverse", reg_only=True)
    _, _, reg_logits, _, _ = LOSS_FN(teacher_topk_logits=logits, **kw)
    _, _, reg_logp, _, _ = LOSS_FN(teacher_topk_logits=logits - shift, **kw)
    assert torch.allclose(reg_logits, reg_logp, atol=1e-5)


def test_reverse_topk_sgd_decreases():
    c = _common()
    stu = torch.randn(BS, L, K, requires_grad=True)
    tlog = torch.randn(BS, L, K)
    opt = torch.optim.SGD([stu], lr=0.5)
    first = last = None
    for step in range(50):
        opt.zero_grad()
        _, _, reg, _, _ = LOSS_FN(
            old_log_prob=c["old_lp"], log_prob=c["old_lp"], advantages=c["adv"], eos_mask=c["eos"],
            teacher_ids_log_probs=c["t_ids_lp"], cliprange=0.2, teacher_coef=1.0,
            teacher_topk_logits=tlog, student_teacher_topk_log_probs=stu,
            kl_direction="reverse", reg_only=True,
        )
        reg.backward(); opt.step()
        first = reg.item() if step == 0 else first
        last = reg.item()
    assert last < first


# --------------------------------------------------------------------------- #
# reverse pointwise K2 / K3 (both >= 0, pathwise via log_prob, teacher detached)
# --------------------------------------------------------------------------- #
def _reverse_pointwise(kd):
    c = _common()
    lp = c["old_lp"].clone().requires_grad_(True)
    teacher_lp = torch.randn(BS, L)
    total, pg, reg, _, _ = LOSS_FN(
        old_log_prob=c["old_lp"], log_prob=lp, advantages=c["adv"], eos_mask=c["eos"],
        teacher_ids_log_probs=c["t_ids_lp"], cliprange=0.2, teacher_coef=COEF,
        teacher_log_prob=teacher_lp, kl_direction=kd, reg_only=True,
    )
    total.backward()
    assert torch.isfinite(total)
    assert reg.item() >= -1e-6                                      # k2, k3 are >= 0
    assert torch.allclose(total, COEF * reg)
    assert lp.grad is not None and torch.isfinite(lp.grad).all()    # pathwise into log_prob
    assert teacher_lp.grad is None                                  # teacher detached


def test_reverse_k2():
    _reverse_pointwise("reverse_k2")


def test_reverse_k3():
    _reverse_pointwise("reverse_k3")


# --------------------------------------------------------------------------- #
# reverse K1 (score-function surrogate) == the retired compute_token_on_rkl_reg_loss.
# Its VALUE has no KL meaning; only its gradient = ∇ D_KL(π_θ||π_T). Unlike k2/k3 it
# keeps the policy-sampling gradient (B) and need not be >= 0.
# --------------------------------------------------------------------------- #
def test_reverse_k1_equals_retired_rkl_reg():
    c = _common()
    lp = c["old_lp"].clone().requires_grad_(True)
    teacher_lp = torch.randn(BS, L)                       # π_T at student tokens (no grad)

    total, pg, reg, _, _ = LOSS_FN(
        old_log_prob=c["old_lp"], log_prob=lp, advantages=c["adv"], eos_mask=c["eos"],
        teacher_ids_log_probs=c["t_ids_lp"], cliprange=0.2, teacher_coef=COEF,
        teacher_log_prob=teacher_lp, kl_direction="reverse_k1", reg_only=True,
    )
    total.backward()

    # (1) value == old surrogate mean( stop_grad(logρ) · logπ_θ )
    w = (lp.detach() - teacher_lp)                         # logρ, detached weight
    ref_reg = _masked_mean(w * lp.detach(), c["eos"])
    assert torch.allclose(reg.detach(), ref_reg, atol=1e-6)
    assert torch.allclose(total.detach(), COEF * ref_reg, atol=1e-6)   # reg_only drops pg_loss

    # (2) gradient == score-function ∇ = coef · logρ · ∇logπ_θ  (weight detached, no "+1")
    expected_grad = COEF * w * c["eos"] / (c["eos"].sum() + 1e-8)
    assert lp.grad is not None and torch.allclose(lp.grad, expected_grad, atol=1e-6)
    assert teacher_lp.grad is None                          # teacher detached (score-function weight)


def test_reverse_k1_mixed_with_grpo():
    """reg_only=False ⇒ total == pg_loss + coef·reg (adds GRPO result reward back)."""
    c = _common()
    lp = c["old_lp"].clone().requires_grad_(True)
    teacher_lp = torch.randn(BS, L)
    total, pg, reg, _, _ = LOSS_FN(
        old_log_prob=c["old_lp"], log_prob=lp, advantages=c["adv"], eos_mask=c["eos"],
        teacher_ids_log_probs=c["t_ids_lp"], cliprange=0.2, teacher_coef=COEF,
        teacher_log_prob=teacher_lp, kl_direction="reverse_k1", reg_only=False,
    )
    total.backward()
    assert torch.isfinite(total)
    assert torch.allclose(total, pg + COEF * reg)
    assert lp.grad is not None and torch.isfinite(lp.grad).all()


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
