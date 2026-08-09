#!/usr/bin/env python3
"""ポリシー設定の自動チューニング・ハーネス（Track 1 example タスク）

`submission_template/policy_server.py` の MyPolicy は、推論時の挙動を多数の
設定（時間方向アンサンブル・TTA・不確実性ダンピング・速度スケール等）で
制御する。それらは環境変数 ``MYPOLICY_<名前>`` で上書きできる。

このスクリプトは、設定の候補を1つずつサーバーに渡して実際に評価パイプラインを
回し、**成功率・平均ステップ・所要時間を実測してランキング**する。勘で置いた
初期値を、example タスク上の実測で最適化するための道具である。

前提:
    bash setup.sh && source env.sh          # キットの環境
    # model_weights/ に SmolVLA 重みを配置済みであること

使い方:
    # 既定: baseline + 主要ノブの一因子ずつ(OFAT)を、指定タスクで評価
    python tune.py --tasks <task_id> [<task_id> ...] --n-episodes 5 --max-steps 300

    # どの設定を試すかだけ確認（サーバーは起動しない）
    python tune.py --list

    # 全タスク・全組合せ（重いので注意）ではなく、対象ノブを絞る
    python tune.py --tasks <id> --knobs UNCERTAINTY_DAMPING SPEED_SCALE

結果は results_tuning/tuning_<timestamp>.json に保存し、標準出力にも表を出す。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SERVER = ROOT / "submission_template" / "policy_server.py"

# 探索するノブと候補値。先頭が「基準値寄り」。OFAT ではここから1つずつ振る。
SWEEP: dict[str, list] = {
    "TEMPORAL_ENSEMBLE": [True, False],
    "REPLAN_INTERVAL": [2, 1, 4],
    "ENSEMBLE_DECAY": [0.1, 0.05, 0.2],
    "TTA_VIEWS": [2, 1, 3],
    "UNCERTAINTY_DAMPING": [6.0, 0.0, 3.0, 12.0],
    "SPEED_SCALE": [1.0, 0.8, 0.6],
    "MAX_POS_DELTA": [1.0, 0.5],
    "ACTION_EMA": [0.0, 0.3],
    "GRIPPER_MARGIN": [0.0, 0.1],
}


def build_configs(knobs: list[str] | None) -> list[dict]:
    """baseline（空 dict = コード既定値）＋ 一因子ずつの変化(OFAT)を返す。"""
    selected = knobs or list(SWEEP.keys())
    configs: list[dict] = [{}]  # baseline
    seen = {frozenset()}
    for knob in selected:
        if knob not in SWEEP:
            print(f"[tune] 未知のノブを無視: {knob}")
            continue
        for value in SWEEP[knob][1:]:  # [0] は基準なので baseline と重複。除く
            cfg = {knob: value}
            key = frozenset(cfg.items())
            if key not in seen:
                seen.add(key)
                configs.append(cfg)
    return configs


def config_to_env(cfg: dict) -> dict:
    env = {}
    for name, value in cfg.items():
        if isinstance(value, bool):
            env[f"MYPOLICY_{name}"] = "1" if value else "0"
        else:
            env[f"MYPOLICY_{name}"] = str(value)
    return env


def config_label(cfg: dict) -> str:
    if not cfg:
        return "baseline"
    return ",".join(f"{k}={v}" for k, v in cfg.items())


def wait_for_health(url: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:  # noqa: BLE001 - まだ起動中
            time.sleep(1.0)
    return False


def run_one(
    cfg: dict,
    port: int,
    tasks: list[str],
    n_episodes: int,
    max_steps: int,
    server_timeout: float,
) -> dict:
    """1 設定でサーバーを起動→評価→結果を集計して返す。"""
    url = f"http://localhost:{port}"
    out_dir = ROOT / "results_tuning" / "_raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    server_env = os.environ.copy()
    server_env.update(config_to_env(cfg))

    server_proc = subprocess.Popen(
        [sys.executable, str(SERVER), "--port", str(port)],
        env=server_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    try:
        if not wait_for_health(url, server_timeout):
            return {"config": cfg, "error": "server did not become healthy"}

        cmd = [
            sys.executable, "-m", "pipeline",
            "--server-url", url,
            "--track", "track1",
            "--n-episodes", str(n_episodes),
            "--max-steps", str(max_steps),
            "--output-dir", str(out_dir),
        ]
        if tasks:
            cmd += ["--tasks", *tasks]

        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            return {
                "config": cfg,
                "error": f"pipeline exited {proc.returncode}",
                "stderr_tail": proc.stderr[-2000:],
            }

        result_path = out_dir / f"server_{port}.json"
        if not result_path.is_file():
            return {"config": cfg, "error": "result json not found"}
        data = json.loads(result_path.read_text(encoding="utf-8"))
        return summarize(cfg, data, elapsed)
    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server_proc.kill()


def summarize(cfg: dict, data: dict, elapsed: float) -> dict:
    tracks = data.get("tracks", [])
    mean_sr = 0.0
    per_task = []
    for ts in tracks:
        metrics = ts.get("metrics", {})
        mean_sr = float(metrics.get("mean_success_rate", ts.get("overall_score", 0.0)))
        for task in ts.get("tasks", []):
            per_task.append(
                {"task": task.get("task_name"), "success_rate": task.get("success_rate")}
            )
    return {
        "config": cfg,
        "mean_success_rate": mean_sr,
        "elapsed_sec": elapsed,
        "per_task": per_task,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ポリシー設定の自動チューニング")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="評価する example タスク名（省略時は全タスク）")
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--server-timeout", type=float, default=180.0,
                        help="サーバー起動（モデルロード含む）待ちの上限秒")
    parser.add_argument("--knobs", nargs="+", default=None,
                        help="振るノブを限定（省略時は全ノブを OFAT）")
    parser.add_argument("--list", action="store_true",
                        help="試す設定を表示して終了（評価しない）")
    args = parser.parse_args()

    configs = build_configs(args.knobs)

    if args.list:
        print(f"試行数: {len(configs)}")
        for i, cfg in enumerate(configs):
            print(f"  [{i:2d}] {config_label(cfg)}")
        return

    print(f"[tune] {len(configs)} 設定を評価します "
          f"(tasks={args.tasks or 'ALL'}, n_episodes={args.n_episodes}, "
          f"max_steps={args.max_steps})")

    results = []
    for i, cfg in enumerate(configs):
        print(f"\n[tune] ({i + 1}/{len(configs)}) {config_label(cfg)} ...")
        res = run_one(
            cfg, args.port, args.tasks or [],
            args.n_episodes, args.max_steps, args.server_timeout,
        )
        if "error" in res:
            print(f"    ERROR: {res['error']}")
        else:
            print(f"    成功率={res['mean_success_rate'] * 100:.1f}%  "
                  f"({res['elapsed_sec']:.0f}s)")
        results.append(res)

    # ランキング（成功率降順、同率は所要時間昇順）
    ok = [r for r in results if "error" not in r]
    ok.sort(key=lambda r: (-r["mean_success_rate"], r["elapsed_sec"]))

    print("\n" + "=" * 70)
    print("ランキング（成功率降順）")
    print("=" * 70)
    for rank, r in enumerate(ok, 1):
        print(f"{rank:2d}. {r['mean_success_rate'] * 100:5.1f}%  "
              f"{r['elapsed_sec']:5.0f}s  {config_label(r['config'])}")
    errs = [r for r in results if "error" in r]
    if errs:
        print("\n-- 失敗した設定 --")
        for r in errs:
            print(f"   {config_label(r['config'])}: {r['error']}")

    out_dir = ROOT / "results_tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"tuning_{stamp}.json"
    out_path.write_text(
        json.dumps(
            {"args": vars(args), "results": results, "ranking": ok},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n結果を保存: {out_path}")
    if ok:
        best = ok[0]
        print(f"\n最良設定: {config_label(best['config'])} "
              f"(成功率 {best['mean_success_rate'] * 100:.1f}%)")
        print("→ この設定を MyPolicy のクラス変数に反映するか、"
              "サーバー起動時に環境変数で固定してください。")


if __name__ == "__main__":
    main()
