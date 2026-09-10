"""Command line interface: ``tdautobpm``.

Subcommands are dispatched before importing anything heavy, so ``doctor`` works even
in an environment where torch is missing - which is exactly when you need it.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import List, Optional

from . import __version__


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def cmd_analyze(args) -> int:
    from .engine import analyze_file

    paths: List[str] = []
    for pattern in args.paths:
        hits = sorted(glob.glob(os.path.expanduser(pattern)))
        paths.extend(hits or [os.path.expanduser(pattern)])

    results, failed = [], 0
    for path in paths:
        try:
            r = analyze_file(
                path,
                checkpoint_path=args.checkpoint,
                device=args.device,
                accumulate=not args.streaming,
                update_hz=args.update_hz,
            )
        except Exception as exc:
            failed += 1
            print(f"{path}: error: {exc}", file=sys.stderr)
            continue

        results.append(r)
        if not args.json:
            print(f"{r['bpm']:7.2f} BPM  conf {r['confidence']:.3f}  "
                  f"{r['duration_s']:6.1f}s  {os.path.basename(r['path'])}")

    if args.json:
        json.dump(results, sys.stdout, indent=2)
        sys.stdout.write("\n")

    return 1 if failed and not results else 0


# ---------------------------------------------------------------------------
# listen
# ---------------------------------------------------------------------------


def cmd_listen(args) -> int:
    try:
        import sounddevice as sd
    except ImportError:
        print(
            "sounddevice is not installed. Install the live extra:\n"
            "  uv pip install -e '.[live]'",
            file=sys.stderr,
        )
        return 2

    import queue

    from .engine import make_predictor

    device = args.device_index
    info = sd.query_devices(device, "input") if device is not None else sd.query_devices(
        kind="input"
    )
    sr = int(args.rate or info["default_samplerate"])
    channels = 1 if args.mono else min(2, int(info["max_input_channels"]) or 1)

    print(f"listening on: {info['name']}  ({sr} Hz, {channels} ch)", file=sys.stderr)

    pred = make_predictor(
        args.checkpoint,
        device=args.device,
        input_sample_rate=sr,
        update_hz=args.update_hz,
        estimate=args.estimate,
        reset_seconds=None if args.accumulate else args.reset_seconds,
        lock_confidence=args.lock_confidence,
    )

    q: "queue.Queue" = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    try:
        with sd.InputStream(
            device=device, channels=channels, samplerate=sr,
            blocksize=args.blocksize, dtype="float32", callback=callback,
        ):
            while True:
                block = q.get()
                mono = block.mean(axis=1) if block.ndim > 1 else block
                out = pred.push_audio(mono)
                if out is None:
                    continue
                bpm, conf, win = out
                if conf < args.min_confidence:
                    continue
                if args.json:
                    print(json.dumps({"bpm": bpm, "confidence": conf, "window_s": win}),
                          flush=True)
                else:
                    bar = "#" * int(conf * 40)
                    print(f"\r{bpm:7.2f} BPM  conf {conf:.3f} {bar:<40}", end="", flush=True)
    except KeyboardInterrupt:
        print(file=sys.stderr)
    return 0


def cmd_devices(args) -> int:
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice is not installed. Try: uv pip install -e '.[live]'",
              file=sys.stderr)
        return 2

    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"{i:3d}  {d['name']}  ({d['max_input_channels']} in, "
                  f"{int(d['default_samplerate'])} Hz)")
    return 0


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


def cmd_serve(args) -> int:
    from .sidecar import main as sidecar_main

    argv = []
    if args.socket:
        argv += ["--socket", args.socket]
    if args.port:
        argv += ["--port", str(args.port)]
    argv += [
        "--device", args.device,
        "--sample-rate", str(args.rate),
        "--update-hz", str(args.update_hz),
        "--estimate", args.estimate,
    ]
    if args.checkpoint:
        argv += ["--checkpoint", args.checkpoint]
    if args.verbose:
        argv += ["-v"]
    return sidecar_main(argv)


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def _touchdesigner_pythons() -> List[str]:
    """Interpreters shipped inside installed TouchDesigner apps."""
    found = []
    roots = ["/Applications", os.path.expanduser("~/Applications"),
             r"C:\Program Files\Derivative"]
    for root in roots:
        if not os.path.isdir(root):
            continue
        for entry in sorted(os.listdir(root)):
            if "touchdesigner" not in entry.lower():
                continue
            base = os.path.join(root, entry)
            found.extend(sorted(glob.glob(
                os.path.join(base, "Contents/Frameworks/Python.framework/Versions/*/bin/python3.*")
            )))
            for exe in ("python.exe", "python3.exe"):
                p = os.path.join(base, exe)
                if os.path.isfile(p):
                    found.append(p)
    # Drop the -config / -intel64 helper binaries.
    return [f for f in found if not f.endswith(("-config", "-intel64"))]


def cmd_doctor(args) -> int:
    from . import envresolve as E

    print(f"tdautobpm {__version__}")
    print(f"running in: {sys.executable}")
    host = E.host_signature()
    print(f"host python: {host['implementation']} {host['version']} {host['machine']} "
          f"({host['abi_tag']})")
    print()

    seen, candidates = set(), []

    def add(origin, spec):
        python = E.python_from_spec(spec)
        if python and python not in seen:
            seen.add(python)
            candidates.append((origin, python))

    for origin, spec in E.candidates(args.env, args.project_root):
        add(origin, spec)
    for spec in args.also or []:
        add("requested", spec)
    for td in _touchdesigner_pythons():
        add("TouchDesigner bundle", td)
    for envs_dir in E._conda_env_dirs():
        for name in sorted(os.listdir(envs_dir))[:40]:
            add(f"conda:{name}", os.path.join(envs_dir, name))

    if not candidates:
        print("No candidate environments found.")
        return 1

    usable_inprocess, usable_sidecar = [], []

    for origin, python in candidates:
        info = E.probe(python, origin=origin)
        print(f"[{origin}]")
        print("  " + info.describe().replace("\n", "\n  "))

        if info.library_validation_blocked:
            print("  !! macOS library validation blocks third-party binaries here.")
            print("     This interpreter can only load them from inside the signed")
            print("     TouchDesigner app, so it cannot be used from a terminal.")
            print("     Use it for in-process mode only, or point at a normal venv/conda env.")
        else:
            inp = E.inprocess_incompatibility(info)
            side = E.sidecar_incompatibility(info)
            print(f"  in-process: {'yes' if inp is None else 'no  - ' + inp}")
            print(f"  sidecar:    {'yes' if side is None else 'no  - ' + side}")
            if inp is None:
                usable_inprocess.append(python)
            if side is None:
                usable_sidecar.append(python)
        print()

    print("summary")
    print(f"  in-process capable: {len(usable_inprocess)}")
    for p in usable_inprocess:
        print(f"    {p}")
    print(f"  sidecar capable:    {len(usable_sidecar)}")
    for p in usable_sidecar:
        print(f"    {p}")

    if not usable_sidecar and not usable_inprocess:
        print()
        print("Nothing usable yet. Create an environment with:")
        print("  uv venv --python 3.11 && uv pip install -e '.[live]'")
        return 1
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="tdautobpm", description=__doc__)
    ap.add_argument("--version", action="version", version=f"tdautobpm {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    common = dict(checkpoint=None, device="cpu")

    p = sub.add_parser("analyze", help="estimate the tempo of audio files")
    p.add_argument("paths", nargs="+", help="audio files or glob patterns")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--checkpoint")
    p.add_argument("--update-hz", type=float, default=4.0)
    p.add_argument("--streaming", action="store_true",
                   help="reset the posterior periodically, as live detection does")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("listen", help="detect tempo from a live audio input")
    p.add_argument("--device-index", type=int, help="input device (see `tdautobpm devices`)")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"],
                   help="torch device")
    p.add_argument("--rate", type=int, help="sample rate; defaults to the device's")
    p.add_argument("--mono", action="store_true")
    p.add_argument("--blocksize", type=int, default=2048)
    p.add_argument("--checkpoint")
    p.add_argument("--update-hz", type=float, default=4.0)
    p.add_argument("--estimate", default="local_mean",
                   choices=["local_mean", "mode", "mean", "median"])
    p.add_argument("--reset-seconds", type=float, default=5.0)
    p.add_argument("--accumulate", action="store_true",
                   help="never reset the posterior")
    p.add_argument("--lock-confidence", type=float, default=0.0)
    p.add_argument("--min-confidence", type=float, default=0.0)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_listen)

    p = sub.add_parser("devices", help="list audio input devices")
    p.set_defaults(func=cmd_devices)

    p = sub.add_parser("serve", help="run the sidecar server")
    p.add_argument("--socket")
    p.add_argument("--port", type=int)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--rate", type=int, default=44100)
    p.add_argument("--update-hz", type=float, default=4.0)
    p.add_argument("--estimate", default="local_mean",
                   choices=["local_mean", "mode", "mean", "median"])
    p.add_argument("--checkpoint")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("doctor", help="diagnose Python environments")
    p.add_argument("--env", help="an environment path, interpreter, or conda env name")
    p.add_argument("--also", action="append", help="additional environment to check")
    p.add_argument("--project-root")
    p.set_defaults(func=cmd_doctor)

    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
