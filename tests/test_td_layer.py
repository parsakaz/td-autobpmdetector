"""The TouchDesigner-facing layer, exercised without TouchDesigner.

The component itself cannot be built outside TD, but the parts that go wrong silently
can be tested here: locating the checkout, and keeping the module importable by an
interpreter that has no torch.

Both `AutoBpmExt.py` and `build_component.py` once relied on `__file__`, which is wrong
in opposite ways. `build_component.py` is run via `exec(open(p).read())`, where
`__file__` is whatever the calling namespace had - in TouchDesigner's textport, a path
inside the application bundle. `AutoBpmExt.py` is stored as a Text DAT *inside* the
.tox and has no file on disk at all. So neither may trust it.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

TD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "touchdesigner")
REPO = os.path.dirname(TD_DIR)


def load_extension_module():
    """Exec AutoBpmExt.py the way a Text DAT would: no __file__, no TD globals."""
    with open(os.path.join(TD_DIR, "AutoBpmExt.py")) as f:
        source = f.read()
    namespace = {"__name__": "AutoBpmExt"}
    exec(compile(source, "<TextDAT>", "exec"), namespace)
    return namespace


@pytest.fixture
def ext():
    return load_extension_module()


class TestExtensionModule:
    def test_imports_without_file_or_td(self, ext):
        assert "AutoBpm" in ext
        assert ext["CHANNELS"] == ("bpm", "confidence", "beat", "phase")

    @pytest.mark.parametrize("filename", ["AutoBpmExt.py", "build_component.py"])
    def test_module_scope_imports_are_stdlib_only(self, filename):
        """These run in TouchDesigner's interpreter, which has no torch or numpy.

        Checked against the parsed module body rather than the text, so prose in the
        docstrings does not count as an import.
        """
        import ast

        with open(os.path.join(TD_DIR, filename)) as f:
            tree = ast.parse(f.read())

        imported = set()
        for node in tree.body:  # module scope only - nested imports are fine
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
            elif isinstance(node, ast.Try):
                for sub in node.body:
                    if isinstance(sub, ast.Import):
                        imported.update(a.name.split(".")[0] for a in sub.names)

        allowed = {"os", "sys", "queue", "threading", "time", "traceback", "td",
                   "__future__"}
        assert imported <= allowed, f"unexpected module-scope imports: {imported - allowed}"

    def test_finds_repo_from_an_explicit_hint(self, ext):
        assert ext["find_repo"](REPO) == REPO

    def test_expands_tilde_and_variables_in_the_hint(self, ext, monkeypatch):
        monkeypatch.setenv("TDBPM_TEST_REPO", REPO)
        assert ext["find_repo"]("$TDBPM_TEST_REPO") == REPO

    def test_finds_repo_from_the_environment(self, ext, monkeypatch):
        monkeypatch.setenv("TDAUTOBPM_REPO", REPO)
        assert ext["find_repo"]() == REPO

    def test_walks_up_from_a_nested_directory(self, ext, monkeypatch):
        monkeypatch.delenv("TDAUTOBPM_REPO", raising=False)
        monkeypatch.chdir(os.path.join(REPO, "src", "tdautobpm"))
        assert ext["find_repo"]() == REPO

    def test_ignores_a_hint_that_is_not_a_checkout(self, ext, monkeypatch):
        """A wrong hint must not win over a real checkout found by walking up."""
        monkeypatch.delenv("TDAUTOBPM_REPO", raising=False)
        monkeypatch.chdir(REPO)
        assert ext["find_repo"]("/Applications/TouchDesigner.app/Contents") == REPO

    def test_error_names_both_escape_hatches(self, ext, monkeypatch, tmp_path):
        monkeypatch.delenv("TDAUTOBPM_REPO", raising=False)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(RuntimeError) as excinfo:
            ext["find_repo"]()
        message = str(excinfo.value)
        assert "Repo Path" in message
        assert "TDAUTOBPM_REPO" in message

    def test_load_support_returns_stdlib_only_helpers(self, ext):
        envresolve, client_cls = ext["load_support"](REPO)
        assert envresolve.__name__ == "tdautobpm.envresolve"
        assert client_cls.__name__ == "SidecarClient"


class TestBuildScript:
    """`build_component.py` resolves its own location at import time."""

    def _exec(self, namespace=None):
        # Stub `td` so the module-scope guard passes outside TouchDesigner.
        stub = types.ModuleType("td")
        saved = sys.modules.get("td")
        sys.modules["td"] = stub
        try:
            with open(os.path.join(TD_DIR, "build_component.py")) as f:
                source = f.read()
            # Strip the trailing main() call; it needs a real TouchDesigner.
            source = source.rsplit("main()", 1)[0]
            ns = {"__name__": "build_component"}
            ns.update(namespace or {})
            exec(compile(source, "<build>", "exec"), ns)
            return ns
        finally:
            if saved is None:
                del sys.modules["td"]
            else:
                sys.modules["td"] = saved

    def test_locates_itself_from_an_explicit_file(self):
        path = os.path.join(TD_DIR, "build_component.py")
        ns = self._exec({"__file__": path})
        assert ns["HERE"] == TD_DIR
        assert ns["REPO"] == REPO

    def test_locates_itself_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("TDAUTOBPM_REPO", REPO)
        ns = self._exec({"__file__": "/Applications/TouchDesigner.app/x.py"})
        assert ns["HERE"] == TD_DIR

    def test_a_bundle_path_from_touchdesigner_is_rejected(self, monkeypatch):
        """The exact failure seen in the textport: __file__ pointing into the app."""
        monkeypatch.delenv("TDAUTOBPM_REPO", raising=False)
        monkeypatch.chdir(REPO)
        bogus = "/Applications/TouchDesigner 2.app/Contents/Resources/tfs/x.py"
        ns = self._exec({"__file__": bogus})
        assert ns["HERE"] == TD_DIR  # fell through to the cwd candidate

    def test_error_explains_how_to_pass_the_path(self, monkeypatch, tmp_path):
        monkeypatch.delenv("TDAUTOBPM_REPO", raising=False)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(RuntimeError) as excinfo:
            self._exec({"__file__": str(tmp_path / "x.py")})
        message = str(excinfo.value)
        assert "exec(open(p).read()" in message
        assert "TDAUTOBPM_REPO" in message

    def test_does_not_rely_on_injected_touchdesigner_globals(self):
        """Everything goes through `td.`, so an explicit globals dict still works."""
        with open(os.path.join(TD_DIR, "build_component.py")) as f:
            source = f.read()
        for name in ("baseCOMP", "textDAT", "scriptCHOP", "inCHOP", "outCHOP",
                     "parameterexecuteDAT"):
            assert f"create({name}," not in source, f"bare {name} used"
            assert f'_td("{name}")' in source
