from __future__ import annotations

import importlib
import pickle
import subprocess
import sys
from pathlib import Path


class TestPublicImportSurface:
    def test_root_module_preserves_public_surface_manifest(self) -> None:
        module = importlib.import_module("agent_rt")
        manifest = (
            Path(__file__)
            .with_name("public_import_surface.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        missing = [name for name in manifest if not hasattr(module, name)]
        assert missing == []

    def test_lazy_feature_preserves_root_module_identity_and_pickle(self) -> None:
        module = importlib.import_module("agent_rt")
        plan_step = module.PlanStep(id="one", title="One")
        assert module.PlanStep.__module__ == "agent_rt"
        restored = pickle.loads(pickle.dumps(plan_step))
        assert restored == plan_step
        assert type(restored) is module.PlanStep

    def test_root_import_does_not_eagerly_load_optional_integrations(self) -> None:
        code = """
import importlib
import sys

importlib.import_module("agent_rt")
assert "ext.runtime.optional" not in sys.modules
assert "openai" not in sys.modules
assert "anthropic" not in sys.modules
assert "agent_action_guard" not in sys.modules
"""
        subprocess.run([sys.executable, "-c", code], check=True)
