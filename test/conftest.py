"""Shared pytest setup.

torch._inductor.test_operators matches pytest's test-file pattern, so pytest's
assertion-rewrite import hook can re-execute it inside stubbed-module contexts
(mock.patch.dict on sys.modules), which re-registers its torch.library
namespace and fails. Import it here once so it is cached before any test runs;
the result is order-independent test collection.
"""

try:
    import torch._inductor.test_operators  # noqa: F401
except Exception:
    pass
