"""The TouchDesigner-facing layer, exercised without TouchDesigner.

The component itself cannot be built outside TD, but the parts that go wrong silently
can be tested here: locating the checkout, and keeping the module importable by an
interpreter that has no torch.

Neither `AutoBpmExt.py` nor `build_component.py` can rely on `__file__`, which is wrong
for them in opposite ways. `build_component.py` is run via `exec(open(p).read())`, where
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
        envresolve, client_cls, envsetup = ext["load_support"](REPO)
        assert envresolve.__name__ == "tdautobpm.envresolve"
        assert client_cls.__name__ == "SidecarClient"
        assert envsetup.__name__ == "tdautobpm.envsetup"

    @pytest.mark.parametrize("filename", ["AutoBpmExt.py", "build_component.py"])
    def test_files_are_ascii(self, filename):
        """TouchDesigner's `open()` decodes as ASCII, so the documented
        `exec(open(p).read())` fails on the first non-ASCII character."""
        with open(os.path.join(TD_DIR, filename), "rb") as f:
            f.read().decode("ascii")


class _Dat:
    """A Text DAT holding one of the embedded modules."""

    def __init__(self, name):
        self.path = "/AutoBpm/lib/" + name
        with open(os.path.join(REPO, "src", "tdautobpm", name + ".py")) as f:
            self.text = f.read()


class _Owner:
    """A component carrying the embedded modules, and nothing else."""

    def __init__(self, missing=()):
        self.missing = set(missing)

    def op(self, path):
        name = path.rsplit("/", 1)[-1]
        return None if name in self.missing else _Dat(name)


@pytest.fixture
def unloaded():
    """Forget the embedded package between tests; it lives in sys.modules."""
    def clear():
        for name in [m for m in sys.modules if m.split(".")[0] == "tdautobpm_embedded"]:
            del sys.modules[name]
    clear()
    yield
    clear()


class TestEmbeddedModules:
    """A downloaded .tox has no checkout, so it carries these itself."""

    def test_loads_the_modules_out_of_the_component(self, ext, unloaded):
        envresolve, client_cls, envsetup = ext["load_support"](None, _Owner())
        assert client_cls.__name__ == "SidecarClient"
        assert envsetup.default_env_dir()
        assert envresolve.expand("~") == os.path.expanduser("~")

    def test_the_client_uses_the_embedded_protocol(self, ext, unloaded):
        """`from . import protocol` must not reach for an installed copy."""
        _, client_cls, _ = ext["load_support"](None, _Owner())
        client_module = sys.modules[client_cls.__module__]
        assert client_module.P is sys.modules["tdautobpm_embedded.protocol"]

    def test_an_older_component_says_what_is_missing(self, ext, unloaded):
        with pytest.raises(RuntimeError) as excinfo:
            ext["load_support"](None, _Owner(missing={"envsetup"}))
        assert "lib/envsetup" in str(excinfo.value)
        assert "Repo Path" in str(excinfo.value)

    def test_a_checkout_is_preferred_when_there_is_one(self, ext, unloaded):
        envresolve, _, _ = ext["load_support"](REPO, _Owner())
        assert envresolve.__name__ == "tdautobpm.envresolve"


class _Par:
    def __init__(self, val):
        self.val = val

    def eval(self):
        return self.val


class _Pars:
    """Parameters the way TD exposes them: assigning a value keeps a Par."""

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value if isinstance(value, _Par) else _Par(value))


class _Comp:
    def __init__(self, **values):
        self.par = _Pars()
        for name, value in values.items():
            setattr(self.par, name, value)

    def op(self, name):
        return None


class TestTapTempo:
    def test_needs_two_taps(self, ext):
        assert ext["tap_tempo"]([]) is None
        assert ext["tap_tempo"]([3.0]) is None

    def test_steady_taps(self, ext):
        bpm, confidence = ext["tap_tempo"]([0.0, 0.5, 1.0, 1.5])
        assert bpm == pytest.approx(120.0)
        assert confidence == pytest.approx(1.0)

    def test_two_taps_give_a_tempo_but_little_confidence(self, ext):
        bpm, confidence = ext["tap_tempo"]([0.0, 0.5])
        assert bpm == pytest.approx(120.0)
        assert confidence < 0.5

    def test_a_missed_tap_does_not_drag_the_tempo(self, ext):
        """A plain mean of these intervals would say 96 BPM."""
        bpm, confidence = ext["tap_tempo"]([0.0, 0.5, 1.0, 2.0, 2.5])
        assert bpm == pytest.approx(120.0)
        assert confidence < 1.0

    def _tap(self, ext, monkeypatch, comp, times):
        bpm = ext["AutoBpm"](comp)
        clock = iter(times)
        monkeypatch.setattr(ext["time"], "time", lambda: next(clock))
        for _ in times:
            bpm.Tap()
        return bpm

    def test_taps_set_the_tempo_and_stop_detection(self, ext, monkeypatch):
        comp = _Comp(Active=True, Repopath=REPO)
        bpm = self._tap(ext, monkeypatch, comp, [10.0, 10.5, 11.0, 11.5])
        assert bpm.bpm == pytest.approx(120.0)
        assert comp.par.Active.eval() is False  # so the next estimate cannot win
        assert bpm._mode() == "tap"
        assert bpm.BeatCount.val == 3  # the fourth tap is beat four

    def test_one_tap_only_realigns_the_phase(self, ext, monkeypatch):
        comp = _Comp(Active=True, Repopath=REPO)
        bpm = ext["AutoBpm"](comp)
        bpm.bpm, bpm.phase = 128.0, 0.6
        monkeypatch.setattr(ext["time"], "time", lambda: 5.0)
        bpm.Tap()
        assert bpm.phase == 0.0
        assert bpm.bpm == 128.0
        assert comp.par.Active.eval() is True

    def test_a_held_key_does_not_tap_again(self, ext, monkeypatch):
        """Auto-repeat from holding space arrives far faster than anyone taps."""
        comp = _Comp(Active=False, Repopath=REPO)
        bpm = self._tap(ext, monkeypatch, comp, [0.0, 0.03, 0.06, 0.5])
        assert len(bpm._taps) == 2
        assert bpm.bpm == pytest.approx(120.0)

    def test_tapping_clears_the_multiplier(self, ext, monkeypatch):
        """What was tapped is the tempo wanted out."""
        comp = _Comp(Active=True, Repopath=REPO, Multiplier=2.0)
        bpm = self._tap(ext, monkeypatch, comp, [0.0, 0.5, 1.0])
        assert comp.par.Multiplier.eval() == 1.0
        assert bpm.OutputBpm == pytest.approx(120.0)

    def test_a_pause_starts_a_new_tap_sequence(self, ext, monkeypatch):
        comp = _Comp(Active=False, Repopath=REPO)
        bpm = self._tap(ext, monkeypatch, comp, [0.0, 1.0, 1.5 + 10.0, 2.0 + 10.0])
        assert bpm.bpm == pytest.approx(120.0)  # not dragged by the 60 BPM pair
        assert bpm.BeatCount.val == 1


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


class TestMultiplier:
    def test_output_is_the_estimate_times_the_multiplier(self, ext):
        comp = _Comp(Active=True, Repopath=REPO, Multiplier=2.0)
        bpm = ext["AutoBpm"](comp)
        bpm.bpm = 87.0
        assert bpm.OutputBpm == pytest.approx(174.0)

    def test_half_and_double_step_by_octaves_within_bounds(self, ext):
        comp = _Comp(Active=True, Repopath=REPO, Multiplier=1.0)
        bpm = ext["AutoBpm"](comp)
        seen = []
        for step in ("Double", "Double", "Double", "Half", "Half", "Half", "Half",
                     "Half"):
            getattr(bpm, step)()
            seen.append(comp.par.Multiplier.eval())
        assert seen == [2.0, 4.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.25]

    def test_phase_runs_at_the_output_tempo(self, ext):
        comp = _Comp(Active=True, Repopath=REPO, Multiplier=2.0)
        bpm = ext["AutoBpm"](comp)
        bpm.bpm = 60.0  # one beat a second, two with the multiplier
        bpm._advance_phase(0.25)
        assert bpm.phase == pytest.approx(0.5)


class TestConfigure:
    def test_range_is_part_of_the_settings(self, ext):
        comp = _Comp(Active=True, Repopath=REPO, Rangemin=160.0, Rangemax=180.0)
        settings = ext["AutoBpm"](comp)._settings()
        assert (settings["range_min"], settings["range_max"]) == (160.0, 180.0)

    def test_forwards_settings_to_a_running_detector(self, ext):
        comp = _Comp(Active=True, Repopath=REPO, Rangemin=160.0, Rangemax=180.0)
        bpm = ext["AutoBpm"](comp)
        received = {}

        class Detector:
            def configure(self, **settings):
                received.update(settings)

        bpm.detector = Detector()
        bpm.Configure()
        assert received["range_min"] == 160.0

    def test_does_nothing_without_a_detector(self, ext):
        ext["AutoBpm"](_Comp(Active=True, Repopath=REPO)).Configure()


class TestPresets:
    """presets.csv is edited by hand, in whatever spreadsheet app is to hand."""

    def test_the_shipped_file_is_clean(self, ext):
        with open(os.path.join(TD_DIR, "presets.csv"), encoding="utf-8") as f:
            presets, problems = ext["read_presets"](f.read())
        assert problems == []
        assert presets[0] == ("Any", 60.0, 240.0)
        assert ("Drum & Bass", 165.0, 180.0) in presets

    def test_notes_header_and_blank_lines_are_skipped(self, ext):
        text = "# a note, with commas\n\nName,Lowest BPM,Highest BPM\nHouse,120,128\n"
        assert ext["read_presets"](text) == ([("House", 120.0, 128.0)], [])

    def test_excel_in_europe_semicolons_and_decimal_commas(self, ext):
        text = "\ufeffName;Lowest BPM;Highest BPM\nDrum & Bass;165;180\nOdd;99,5;101\n"
        presets, problems = ext["read_presets"](text)
        assert presets == [("Drum & Bass", 165.0, 180.0), ("Odd", 99.5, 101.0)]
        assert problems == []

    def test_tabs_quotes_and_reversed_tempos(self, ext):
        text = 'Name\tLow\tHigh\n"Reggaeton, Dembow"\t102\t88\n'
        assert ext["read_presets"](text)[0] == [("Reggaeton, Dembow", 88.0, 102.0)]

    def test_a_line_typed_with_another_separator_still_counts(self, ext):
        text = "Name,Low,High\nHouse,120,128\nTypo;100;110\n"
        presets, problems = ext["read_presets"](text)
        assert ("Typo", 100.0, 110.0) in presets
        assert problems == []

    def test_bad_lines_are_reported_by_number_not_dropped_silently(self, ext):
        text = "Name,Low,High\nHouse,120,128\nBroken,fast,faster\n,100,110\nFlat,120,120\n"
        presets, problems = ext["read_presets"](text)
        assert presets == [("House", 120.0, 128.0)]
        assert [p.split(":")[0] for p in problems] == ["line 3", "line 4", "line 5"]

    def test_an_empty_file_has_no_presets(self, ext):
        assert ext["read_presets"]("") == ([], [])


class TestHold:
    def _comp(self, **values):
        values.setdefault("Active", True)
        values.setdefault("Repopath", REPO)
        values.setdefault("Autosyncconfidence", 0.5)
        values.setdefault("Hold", True)
        return _Comp(**values)

    def test_a_doubtful_estimate_does_not_replace_a_confident_one(self, ext):
        bpm = ext["AutoBpm"](self._comp())
        bpm._take_estimate(174.0, 0.8)
        bpm._take_estimate(131.0, 0.1)  # the breakdown
        assert bpm.bpm == 174.0
        assert bpm.RawBpm.val == 131.0  # the panel still shows what it hears
        assert bpm.confidence == 0.1
        assert bpm._mode() == "hold"
        bpm._take_estimate(175.0, 0.7)  # the drop
        assert bpm.bpm == 175.0
        assert bpm._mode() == "auto"

    def test_passes_through_until_there_is_something_to_hold(self, ext):
        bpm = ext["AutoBpm"](self._comp())
        bpm._take_estimate(120.0, 0.1)
        bpm._take_estimate(122.0, 0.2)
        assert bpm.bpm == 122.0
        assert bpm._mode() == "auto"

    def test_off_follows_every_estimate(self, ext):
        bpm = ext["AutoBpm"](self._comp(Hold=False))
        bpm._take_estimate(174.0, 0.8)
        bpm._take_estimate(131.0, 0.1)
        assert bpm.bpm == 131.0

    def test_reset_forgets_the_held_tempo(self, ext):
        bpm = ext["AutoBpm"](self._comp())
        bpm._take_estimate(174.0, 0.8)
        bpm.Reset()
        bpm._take_estimate(131.0, 0.1)
        assert bpm.bpm == 131.0

    def test_a_range_that_excludes_the_held_tempo_releases_it(self, ext):
        comp = self._comp(Rangemin=60.0, Rangemax=240.0)
        bpm = ext["AutoBpm"](comp)
        bpm._take_estimate(174.0, 0.8)
        comp.par.Rangemin, comp.par.Rangemax = 120.0, 135.0
        bpm.Configure()
        bpm._take_estimate(128.0, 0.2)
        assert bpm.bpm == 128.0

    def test_a_tapped_tempo_holds_until_detection_is_sure(self, ext, monkeypatch):
        comp = self._comp()
        bpm = ext["AutoBpm"](comp)
        clock = iter([0.0, 0.5, 1.0])
        monkeypatch.setattr(ext["time"], "time", lambda: next(clock))
        for _ in range(3):
            bpm.Tap()
        comp.par.Active = True  # START
        bpm._take_estimate(90.0, 0.2)
        assert bpm.bpm == pytest.approx(120.0)
