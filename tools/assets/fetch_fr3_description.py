#!/usr/bin/env python3
"""获取 franka_description -> 生成 fr3.urdf(xacro) -> staging 到 franka/。

``franka_description`` 只提供 ``.urdf.xacro``，且 xacro 里的
``$(find franka_description)`` 需要 ``ament_index_python``。本脚本注入一个
最小 ament stub（把该包指向 checkout），因此不需要安装 ROS。

前置：``pip install xacro``；staging 阶段若视觉网格是 ``.dae``，还需
``pip install trimesh pycollada`` 让 setup_fr3 转成 ``.stl``。

用法::

    python tools/assets/fetch_fr3_description.py            # 克隆到 /tmp/franka_description 并 staging
    python tools/assets/fetch_fr3_description.py --checkout ~/franka_description
    python tools/assets/fetch_fr3_description.py --no-hand  # 不带 Franka Hand
    python tools/assets/fetch_fr3_description.py --check    # 只报告 staging 状态
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = "https://github.com/frankarobotics/franka_description"
DEFAULT_CHECKOUT = Path(tempfile.gettempdir()) / "franka_description"


def _load_setup_fr3():
    spec = importlib.util.spec_from_file_location("setup_fr3", HERE / "setup_fr3.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def ensure_checkout(checkout: Path, repo: str, ref: str) -> Path:
    if (checkout / "robots").is_dir():
        print(f"[fetch] 复用 checkout: {checkout}")
        return checkout
    checkout.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] git clone {repo} -> {checkout}")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", ref, repo, str(checkout)],
        check=True,
    )
    return checkout


def _write_ament_stub(checkout: Path) -> Path:
    stub = Path(tempfile.mkdtemp(prefix="fr3_ament_stub_"))
    pkg = stub / "ament_index_python"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "packages.py").write_text(
        "def get_package_share_directory(pkg):\n"
        "    if pkg != 'franka_description':\n"
        "        raise LookupError(pkg)\n"
        f"    return {str(checkout)!r}\n"
    )
    return stub


def generate_urdf(checkout: Path, hand: bool, out: Path) -> Path:
    xacro_file = checkout / "robots" / "fr3" / "fr3.urdf.xacro"
    if not xacro_file.is_file():
        raise FileNotFoundError(f"找不到 xacro: {xacro_file}")
    stub = _write_ament_stub(checkout)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(stub), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    # xacro 没有 __main__，用 console script 或 -c 调 main()。
    console = Path(sys.executable).parent / "xacro"
    if console.is_file():
        cmd = [str(console), str(xacro_file), f"hand:={'true' if hand else 'false'}"]
    else:
        cmd = [
            sys.executable,
            "-c",
            "import sys, xacro; sys.argv = ['xacro'] + sys.argv[1:]; xacro.main()",
            str(xacro_file),
            f"hand:={'true' if hand else 'false'}",
        ]
    print(f"[fetch] 生成 URDF: {out}  (hand={hand})")
    with out.open("w") as fh:
        subprocess.run(cmd, check=True, env=env, stdout=fh)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="获取并 staging FR3 描述")
    parser.add_argument("--checkout", type=Path, default=DEFAULT_CHECKOUT)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--ref", default="main")
    parser.add_argument("--no-hand", action="store_true", help="不带 Franka Hand")
    parser.add_argument("--dest", type=Path, default=None, help="staging 目标（默认 franka/）")
    parser.add_argument("--check", action="store_true", help="只报告 staging 状态")
    parser.add_argument("--force", action="store_true", help="覆盖已有 staging")
    args = parser.parse_args(argv)

    setup = _load_setup_fr3()
    dest = args.dest.resolve() if args.dest else setup.DEST

    if args.check:
        return setup.main(["--check", "--dest", str(dest)])

    if (dest / "urdf" / "fr3.urdf").is_file() and not args.force:
        print(f"[fetch] 已 staged: {dest}（需要重建加 --force）")
        return 0

    try:
        import xacro  # noqa: F401
    except ImportError:
        print("[fetch] 缺少 xacro，先 `pip install xacro`", file=sys.stderr)
        return 2

    checkout = ensure_checkout(args.checkout.resolve(), args.repo, args.ref)
    urdf = generate_urdf(checkout, hand=not args.no_hand, out=checkout / "fr3.generated.urdf")

    print(f"[fetch] staging -> {dest}")
    return setup.main(
        [
            "--source", str(checkout),
            "--source-urdf", str(urdf),
            "--dest", str(dest),
            *(["--force"] if args.force else []),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
