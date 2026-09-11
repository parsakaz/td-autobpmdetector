"""Environment discovery and compatibility rules."""

from __future__ import annotations

import os
import sys

import pytest

from tdautobpm import envresolve as E


class TestExpand:
    def test_expands_tilde(self):
        """A literal '~' would otherwise become a directory *named* '~'. Every
        user-supplied path goes through expand()."""
        assert E.expand("~/envs/foo") == os.path.join(os.path.expanduser("~"), "envs/foo")
        assert "~" not in E.expand("~/envs/foo")

    def test_expands_variables(self, monkeypatch):
        monkeypatch.setenv("TDBPM_TEST_DIR", "/opt/envs")
        assert E.expand("$TDBPM_TEST_DIR/x") == "/opt/envs/x"

    def test_strips_whitespace(self):
        assert E.expand("  /tmp/x  ") == "/tmp/x"

    def test_returns_absolute_paths(self):
        assert os.path.isabs(E.expand("relative/path"))


class TestPythonFromSpec:
    def test_accepts_an_interpreter_path(self):
        assert E.python_from_spec(sys.executable) == sys.executable

    def test_accepts_an_environment_directory(self):
        prefix = sys.prefix
        found = E.python_from_spec(prefix)
        assert found is not None
        assert os.path.isfile(found)

    def test_returns_none_for_nonsense(self):
        assert E.python_from_spec("/nonexistent/environment/xyz") is None

    def test_returns_none_for_empty(self):
        assert E.python_from_spec("") is None
        assert E.python_from_spec("   ") is None

    def test_a_name_is_not_treated_as_a_path(self):
        # Should attempt conda-name resolution and simply find nothing.
        assert E.python_from_spec("definitely-not-a-real-conda-env") is None


class TestProbe:
    def test_probes_the_running_interpreter(self):
        info = E.probe(sys.executable)
        assert info.ok
        assert info.version.startswith("%d.%d" % sys.version_info[:2])
        assert info.modules["torch"] is True
        assert info.complete
        assert not info.missing

    def test_missing_interpreter_is_reported_not_raised(self):
        info = E.probe("/nonexistent/python")
        assert not info.ok
        assert "no such interpreter" in info.error

    def test_site_packages_exist(self):
        info = E.probe(sys.executable)
        paths = info.site_packages()
        assert paths and all(os.path.isdir(p) for p in paths)


class TestCompatibility:
    def test_current_interpreter_is_compatible_with_itself(self):
        info = E.probe(sys.executable)
        assert E.inprocess_incompatibility(info) is None
        assert E.sidecar_incompatibility(info) is None

    def test_minor_version_mismatch_blocks_inprocess_but_not_sidecar(self):
        """Compiled extensions are not portable across CPython minor versions, so
        in-process is refused. A separate process does not care."""
        info = E.probe(sys.executable)
        info.abi_tag = "cp312"
        info.version = "3.12.0"
        info.version_info = (3, 12, 0)

        reason = E.inprocess_incompatibility(info)
        assert reason and "cp312" in reason
        assert E.sidecar_incompatibility(info) is None

    def test_architecture_mismatch_blocks_inprocess(self):
        info = E.probe(sys.executable)
        info.machine = "s390x"
        assert "s390x" in E.inprocess_incompatibility(info)

    def test_missing_packages_block_both(self):
        info = E.probe(sys.executable)
        info.modules = dict(info.modules, torch=False)
        assert "torch" in E.inprocess_incompatibility(info)
        assert "torch" in E.sidecar_incompatibility(info)

    def test_too_old_for_sidecar(self):
        info = E.probe(sys.executable)
        info.version_info = (3, 8, 0)
        info.version = "3.8.0"
        assert "too old" in E.sidecar_incompatibility(info)


class TestResolve:
    def test_explicit_spec_wins(self):
        info = E.resolve(sys.executable)
        assert info.python == sys.executable
        assert info.origin == "explicit"

    def test_env_var_is_honoured(self, monkeypatch):
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        monkeypatch.setenv("TDAUTOBPM_PYTHON", sys.executable)
        assert E.resolve().origin == "TDAUTOBPM_PYTHON"

    def test_failure_message_is_actionable(self, monkeypatch):
        for var in ("TDAUTOBPM_PYTHON", "CONDA_PREFIX", "VIRTUAL_ENV"):
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(E.EnvError) as excinfo:
            E.resolve("/nonexistent/env", project_root="/nonexistent/project")
        message = str(excinfo.value)
        assert "uv venv" in message
        assert "TDAUTOBPM_PYTHON" in message
