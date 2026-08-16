# Vendored from upstream

This directory is a vendored copy of https://github.com/safety-research/assistant-axis
at commit `a98961956072224eaf244eb289d6c01700b63795` (vendored 2026-08-12), tracked
directly in this sprint repo so our modifications persist and clone with it.

Local additions/changes vs upstream:
- `og-paper.md` — markdown copy of the paper (arXiv:2601.10387), not in upstream.
- `assistant_axis/generation.py` — `format_conversation()` honours a new env var
  `ASSISTANT_AXIS_FORCE_USER_CONCAT=1`, which forces persona instructions into the user turn
  instead of a system turn. Needed because the SPP models were trained without a system role:
  upstream's auto-detection can take the system path if the chat template silently renders an
  unsupported system turn, in which case the persona instruction is ignored and the model looks
  like it simply can't role-play. `scripts/check_chat_template.py` reports which path applies.

To diff against upstream later: fetch the repo at the commit above and compare.
