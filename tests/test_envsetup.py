"""Working out where an environment goes and how to build it."""

from __future__ import annotations

import os
import sys

import pytest

from tdautobpm import envsetup as S


class TestLocations:
    def test_the_default_environment_is_per_user_not_per_project(self):
        assert S.default_env_dir().startswith(os.path.expanduser("~"))
        assert S.default_env_dir() == os.path.join(S.app_support_dir(), "env")

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS layout")
    def test_macos_uses_application_support(self):
        assert S.app_support_dir().endswith("Library/Application Support/tdautobpm")

    def test_linux_respects_xdg_data_home(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(os, "name", "posix")
        monkeypatch.setenv("XDG_DATA_HOME", "/tmp/share")
        assert S.app_support_dir() == "/tmp/share/tdautobpm"

    def test_the_interpreter_inside_an_environment(self):
        assert S.env_python("/envs/x") == os.path.join("/envs/x", "bin", "python")


class TestPackage:
    def test_the_wheel_url_names_an_exact_release(self):
        url = S.package_url("0.1.2")
        assert url.endswith("/v0.1.2/tdautobpm-0.1.2-py3-none-any.whl")

    def test_the_filename_is_one_pip_accepts(self):
        """pip rejects a wheel whose filename carries no version."""
        name = S.package_url("1.2.3").rsplit("/", 1)[-1]
        distribution, version, python, abi, platform = name[: -len(".whl")].split("-")
        assert (distribution, version) == ("tdautobpm", "1.2.3")
        assert (python, abi, platform) == ("py3", "none", "any")


class TestPlan:
    def test_extra_pip_arguments_are_passed_through(self):
        """So that an index URL or --no-deps can be put in the Package field."""
        _, install = (argv for _, argv in
                      S.create_steps("/envs/x", "py", "--no-deps /tmp/a.whl"))
        assert install[-2:] == ["--no-deps", "/tmp/a.whl"]

    def test_steps_create_then_install_into_the_new_environment(self):
        steps = S.create_steps("/envs/x", "/usr/bin/python3.12", "WHEEL")
        labels = [label for label, _ in steps]
        create, install = (argv for _, argv in steps)
        assert len(labels) == 2
        assert create == ["/usr/bin/python3.12", "-m", "venv", "/envs/x"]
        assert install[0] == S.env_python("/envs/x")
        assert install[1:4] == ["-m", "pip", "install"]
        assert install[-1] == "WHEEL"

    def test_the_host_interpreter_is_the_last_resort(self):
        """An environment on TouchDesigner's Python only works inside TouchDesigner."""
        found = S.find_base_interpreters(host_python=sys.executable)
        assert found, "the interpreter running the tests should qualify"
        hosts = [i for i, (_, _, is_host) in enumerate(found) if is_host]
        assert all(i == len(found) - 1 for i in hosts)

    def test_candidates_are_usable_interpreters(self):
        for python, version, _ in S.find_base_interpreters():
            assert os.access(python, os.X_OK)
            assert version.startswith("3.")

    def test_the_plan_says_what_and_where(self):
        line = S.describe_plan("/envs/x", "/usr/bin/python3.12", "3.12", False)
        assert "/envs/x" in line and "3.12" in line and "/usr/bin/python3.12" in line
        assert "TouchDesigner" in S.describe_plan("/envs/x", "/tdpy", "3.11", True)
