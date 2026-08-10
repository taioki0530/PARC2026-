"""ポリシーサーバー（提出用テンプレート）

このファイルを編集して、自分のモデルを組み込んでください。
編集が必要なのは MyPolicy クラスの中身だけです。
それ以外のコード（サーバー部分、シリアライゼーション）は変更不可です。

ローカルテスト:
    pip install -r requirements.txt
    python policy_server.py                  # サーバー起動（port 8000）

    # 別ターミナルで評価実行
    python -m pipeline --server-url http://localhost:8000 --dry-run
"""

import argparse
from abc import ABC, abstractmethod

import msgpack
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response


# ============================================================
# ポリシーのインターフェース定義（変更不可）
# MyPolicy が満たすべき get_action() / reset() の仕様を定める。
# ============================================================


class BasePolicy(ABC):
    """ポリシーの基底クラス。get_action() と reset() を実装してください。"""

    @abstractmethod
    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """観測からアクションを推論する。

        Args:
            obs: 環境からの観測。以下のキーが含まれる:
                - "agentview_image": (128, 128, 3) uint8
                - "robot0_eye_in_hand_image": (128, 128, 3) uint8
                - "robot0_joint_pos": (7,) float
                - "robot0_eef_pos": (3,) float
                - "robot0_eef_quat": (4,) float
                - "robot0_gripper_qpos": (2,) float

        Returns:
            action: (7,) float32 — [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        ...

    @abstractmethod
    def reset(self, instruction: str = "") -> None:
        """エピソード開始時に呼ばれる。内部状態をリセットしてください。

        Args:
            instruction: タスクの言語指示（例: "pick up the red mug and place it on the shelf"）
        """
        ...


# ============================================================
# ここを編集する（MyPolicy の中身だけを自分のモデルに置き換える）
# ============================================================


class MyPolicy(BasePolicy):
    """SmolVLA を土台にした、Track 1 の採点軸を狙い撃つ推論時アーキテクチャ。

    ------------------------------------------------------------------
    なぜ「素の VLA」ではないのか — この競技で実際に点になるもの
    ------------------------------------------------------------------
    1. 成功判定が極端に厳しい: ゴール達成 *かつ* 対象外物体の変位が全ステップで
       1mm 以下。素の VLA は物にわずかに触れただけで失格する。
       → 迷ったら減速し、動きを滑らかにして「触れない」ことが本質。
    2. 摂動タスク（背景テクスチャ・照明 L2〜L5）: 単一の前向き推論は分布シフトで崩れる。
       → 見え方をジッタした複数ビューの合議で頑健化する。
    3. 軌道メトリクス（SPARC/jerk）が記録される: ガタつく軌道は不利。
       → 時間方向の重ね合わせと低域通過で滑らかにする。

    これらに対し、SmolVLA を「知覚→意図」のバックボーンとしつつ、その上に
    次の4段を重ねる（学習済み重みだけで完結し、追加学習は不要）:

      (A) 時間方向アンサンブル（ACT 式 temporal ensembling）
          毎ステップ近傍で予測した重複チャンクを、そのステップを指す成分だけ
          指数重みで集約する。新旧の予測が互いを平滑化し、成功率と滑らかさが
          同時に上がる。
      (B) テスト時視覚オーグメンテーション（TTA）
          明度・コントラスト・ガンマを軽くジッタした K 枚のビューでチャンクを
          予測し平均する。テクスチャ/照明の摂動に対する不変性を推論時に獲得する。
      (C) 不確実性適応ダンピング
          TTA ビュー間の予測不一致を「自信のなさ」とみなし、大きいほど並進・回転を
          減速する。曖昧な局面ほど慎重に動き、対象外物体への接触を避ける。
      (D) 衝突配慮の出力整形
          速度スケール・1 ステップあたりのデルタ上限・EMA 低域通過・グリッパの
          ヒステリシスで、オーバーシュートとチャタリングを抑える。

    どの段も設定（下のクラス変数）で強度を変えられ、無効化もできる。API 差異や
    例外が起きても安全なアクションを返し、サーバーを落とさない設計にしている。

      (E) マルチチェックポイント・アンサンブル
          model_weights/ 直下に複数モデルを並べると、全モデル × TTA ビューで
          予測して実座標平均する。異なる fine-tune（例: 通常 / 照明強め / テクスチャ
          強め）を分担させ、単一モデルの穴を埋める。(C) の不確実性はモデル間の
          ばらつきも拾うので、モデルが割れる局面ほど自動で慎重になる。

    設定は全てクラス変数で、さらに環境変数 MYPOLICY_<名前> で上書きできる
    （コード改変なしで tune.py が総当たり探索する。例: MYPOLICY_TTA_VIEWS=3）。

    ------------------------------------------------------------------
    重みの配置（採点環境は外部通信を遮断するため必ず同梱すること）
    ------------------------------------------------------------------
        submission.zip
        ├── policy_server.py
        ├── requirements.txt
        └── model_weights/        # ← save_pretrained() の出力一式をそのまま置く
            ├── config.json
            ├── model.safetensors
            ├── policy_preprocessor.json / .safetensors
            └── policy_postprocessor.json / .safetensors

    # 複数チェックポイントをアンサンブルする場合（任意）:
        └── model_weights/
            ├── base/     { config.json, model.safetensors, ... }
            ├── light/    { config.json, model.safetensors, ... }
            └── texture/  { config.json, model.safetensors, ... }
    # もしくは MYPOLICY_MODEL_DIRS="modelA,modelB" のようにカンマ区切りで指定する。

    観測・アクション仕様（LeRobot の LIBERO 評価ラッパに整合）:
      - agentview → observation.images.image / eye_in_hand → observation.images.image2
      - observation.state(8) = eef_pos(3) + quat→axis-angle(3) + gripper_qpos(2)
      - action(7) = [dx, dy, dz, droll, dpitch, dyaw, gripper]（relative, [-1, 1]）
    """

    # ==== 基本設定 ==========================================================
    MODEL_DIR = "model_weights"       # LeRobot 形式の重みを置くディレクトリ
    FLIP_IMAGES = True                # LIBERO(OffScreenRender)は上下反転で返る
    CAMERA_KEYS = {
        "agentview_image": "observation.images.image",
        "robot0_eye_in_hand_image": "observation.images.image2",
    }

    # ==== (A) 時間方向アンサンブル ==========================================
    TEMPORAL_ENSEMBLE = True          # ACT 式の重ね合わせを有効化
    REPLAN_INTERVAL = 2               # 何ステップごとにチャンクを再予測するか
    ENSEMBLE_DECAY = 0.1              # 過去チャンクの重み = exp(-DECAY * 経過ステップ)

    # ==== (B) テスト時視覚オーグメンテーション ==============================
    TTA_VIEWS = 2                     # 予測に使うビュー数（1 で TTA 無効）
    TTA_BRIGHTNESS = 0.15             # 明度ジッタ幅（±割合）
    TTA_CONTRAST = 0.15               # コントラストジッタ幅（±割合）
    TTA_GAMMA = 0.12                  # ガンマジッタ幅（±割合）

    # ==== (C) 不確実性適応ダンピング ========================================
    UNCERTAINTY_DAMPING = 6.0         # 減速の強さ（0 で無効）。scale=1/(1+k*不一致)

    # ==== (D) 衝突配慮の出力整形 ============================================
    SPEED_SCALE = 1.0                 # 並進・回転デルタの全体スケール
    MAX_POS_DELTA = 1.0               # 1 ステップ並進デルタの各軸上限（<1 で慎重）
    ACTION_EMA = 0.0                  # 最終アクションへの追加低域通過（0 で無効）
    GRIPPER_MARGIN = 0.0              # グリッパのヒステリシス幅（0 で素通し）

    # チャンク行動次元のうち末尾をグリッパとみなす（LIBERO は index 6）
    GRIPPER_IDX = 6

    def __init__(self):
        import os

        import numpy as _np
        import torch

        self._torch = torch
        self._np = _np
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.instruction = ""
        self._rng = _np.random.default_rng(0)

        # 環境変数で全設定を上書きできるようにする（チューニング用。tune.py が使う）。
        # 例: MYPOLICY_TTA_VIEWS=3 MYPOLICY_UNCERTAINTY_DAMPING=8 python policy_server.py
        self._load_overrides_from_env()

        here = os.path.dirname(os.path.abspath(__file__))
        model_dirs = self._discover_model_dirs(here)
        if not model_dirs:
            raise FileNotFoundError(
                f"モデルが見つかりません: {os.path.join(here, self.MODEL_DIR)}. "
                "LeRobot 形式の SmolVLA 重み一式を model_weights/ に配置してください"
                "（複数チェックポイントをアンサンブルする場合は "
                "model_weights/<name>/ 以下に各モデルを置くか、"
                "MYPOLICY_MODEL_DIRS にカンマ区切りで指定する）。"
            )

        # 各モデルを「エンジン」（policy + 前後処理 + chunk API 有無）として保持。
        self.engines: list[dict] = []
        for md in model_dirs:
            self.engines.append(self._load_engine(md))
        print(f"[MyPolicy] {len(self.engines)} モデルをロード: {model_dirs}")

        self.reset("")

    # ------------------------------------------------------------------
    # 設定の環境変数オーバーライド / モデル探索・ロード
    # ------------------------------------------------------------------
    def _load_overrides_from_env(self) -> None:
        import os

        # クラス変数のうち、スカラー設定を MYPOLICY_<NAME> で上書きする。
        overridable = {
            "FLIP_IMAGES": bool,
            "TEMPORAL_ENSEMBLE": bool,
            "REPLAN_INTERVAL": int,
            "ENSEMBLE_DECAY": float,
            "TTA_VIEWS": int,
            "TTA_BRIGHTNESS": float,
            "TTA_CONTRAST": float,
            "TTA_GAMMA": float,
            "UNCERTAINTY_DAMPING": float,
            "SPEED_SCALE": float,
            "MAX_POS_DELTA": float,
            "ACTION_EMA": float,
            "GRIPPER_MARGIN": float,
        }
        for name, caster in overridable.items():
            raw = os.environ.get(f"MYPOLICY_{name}")
            if raw is None:
                continue
            try:
                if caster is bool:
                    val = raw.strip().lower() in ("1", "true", "yes", "on")
                else:
                    val = caster(raw)
                setattr(self, name, val)  # インスタンス属性でクラス変数を隠す
                print(f"[MyPolicy] override {name}={val}")
            except Exception as exc:  # noqa: BLE001
                print(f"[MyPolicy] override {name} 無効値 '{raw}' を無視: {exc}")

    def _discover_model_dirs(self, here: str) -> list[str]:
        import os

        # 明示指定（カンマ区切り、絶対 or here 相対）を最優先。
        env_dirs = os.environ.get("MYPOLICY_MODEL_DIRS")
        if env_dirs:
            dirs = []
            for d in env_dirs.split(","):
                d = d.strip()
                if not d:
                    continue
                dirs.append(d if os.path.isabs(d) else os.path.join(here, d))
            return [d for d in dirs if self._is_model_dir(d)]

        root = os.path.join(here, self.MODEL_DIR)
        if not os.path.isdir(root):
            return []
        # root 自体がモデル一式なら単一モデル。
        if self._is_model_dir(root):
            return [root]
        # そうでなければ、直下のサブディレクトリのうちモデル一式のものを全て使う
        #（マルチチェックポイント・アンサンブル）。
        subs = sorted(
            os.path.join(root, name)
            for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))
        )
        return [d for d in subs if self._is_model_dir(d)]

    @staticmethod
    def _is_model_dir(path: str) -> bool:
        import os

        return os.path.isfile(os.path.join(path, "config.json")) and (
            os.path.isfile(os.path.join(path, "model.safetensors"))
            or os.path.isfile(os.path.join(path, "model.pt"))
            or any(
                n.endswith(".safetensors") for n in os.listdir(path)
            )
        )

    def _load_engine(self, model_dir: str) -> dict:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        policy = SmolVLAPolicy.from_pretrained(model_dir)  # ローカルパス（外部通信なし）
        policy.to(self.device)
        policy.eval()

        pre = post = None
        try:
            from lerobot.processor import make_pre_post_processors

            pre, post = make_pre_post_processors(
                policy.config, pretrained_path=model_dir
            )
        except Exception as exc:  # noqa: BLE001 - 版差異はフォールバックへ
            print(
                f"[MyPolicy] {model_dir}: 前後処理パイプライン未ロード ({exc}). "
                "素通し経路にフォールバックします。"
            )
        return {
            "policy": policy,
            "pre": pre,
            "post": post,
            "has_chunk": hasattr(policy, "predict_action_chunk"),
        }

    @property
    def _has_chunk_api(self) -> bool:
        # 全エンジンが full-chunk 予測に対応しているときのみ時間方向アンサンブルを使う。
        return all(e["has_chunk"] for e in self.engines)

    # ------------------------------------------------------------------
    # エピソード制御
    # ------------------------------------------------------------------
    def reset(self, instruction: str = "") -> None:
        self.instruction = instruction or ""
        self._step = 0
        # (start_step, chunk[H,7], disagreement) のバッファ
        self._chunks: list[tuple[int, "np.ndarray", float]] = []
        self._prev_action = None       # EMA 用
        self._gripper_state = 0.0      # ヒステリシス用
        for eng in getattr(self, "engines", []):
            pol = eng["policy"]
            if hasattr(pol, "reset"):
                try:
                    pol.reset()
                except Exception:  # noqa: BLE001
                    pass

    # ------------------------------------------------------------------
    # 観測 → モデル入力
    # ------------------------------------------------------------------
    @staticmethod
    def _quat_to_axis_angle(quat) -> "np.ndarray":
        # robosuite の quaternion 規約は [x, y, z, w]。
        q = np.asarray(quat, dtype=np.float64).reshape(4)
        norm = np.linalg.norm(q)
        if norm < 1e-8:
            return np.zeros(3, dtype=np.float64)
        q = q / norm
        if q[3] < 0:  # w を正に揃え回転角を [-pi, pi] に
            q = -q
        angle = 2.0 * np.arccos(np.clip(q[3], -1.0, 1.0))
        sin_half = np.sqrt(max(0.0, 1.0 - q[3] * q[3]))
        if sin_half < 1e-8:
            return np.zeros(3, dtype=np.float64)
        return (q[:3] / sin_half * angle).astype(np.float64)

    def _photometric_jitter(self, img01: "np.ndarray") -> "np.ndarray":
        # img01: HWC float32 [0,1]。明度/コントラスト/ガンマを軽く揺らす。
        np_ = self._np
        rng = self._rng
        b = 1.0 + rng.uniform(-self.TTA_BRIGHTNESS, self.TTA_BRIGHTNESS)
        c = 1.0 + rng.uniform(-self.TTA_CONTRAST, self.TTA_CONTRAST)
        g = 1.0 + rng.uniform(-self.TTA_GAMMA, self.TTA_GAMMA)
        out = img01 * b
        mean = out.mean()
        out = (out - mean) * c + mean
        out = np_.clip(out, 0.0, 1.0) ** g
        return np_.clip(out, 0.0, 1.0).astype(np_.float32)

    def _build_batch(self, obs, image_variant: int = 0) -> dict:
        torch = self._torch
        np_ = self._np
        batch: dict = {}

        for raw_key, feat_key in self.CAMERA_KEYS.items():
            img = obs[raw_key]
            if self.FLIP_IMAGES:
                img = img[::-1]
            img01 = np_.ascontiguousarray(img).astype(np_.float32) / 255.0
            if image_variant > 0:  # TTA: 原画像(variant=0)以外はジッタ
                img01 = self._photometric_jitter(img01)
            tensor = torch.from_numpy(np_.ascontiguousarray(img01))
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            batch[feat_key] = tensor.to(self.device)

        eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)[:3]
        axis_angle = self._quat_to_axis_angle(obs["robot0_eef_quat"]).astype(np.float32)
        gripper = np.asarray(
            obs["robot0_gripper_qpos"], dtype=np.float32
        ).reshape(-1)[:2]
        state = np.concatenate([eef_pos, axis_angle, gripper]).astype(np.float32)
        batch["observation.state"] = (
            torch.from_numpy(state).unsqueeze(0).to(self.device)
        )
        batch["task"] = [self.instruction]
        return batch

    # ------------------------------------------------------------------
    # モデル推論（1 ビュー分のチャンクを実座標で返す）
    # ------------------------------------------------------------------
    def _predict_chunk_one(self, obs, image_variant: int, engine: dict) -> "np.ndarray":
        torch = self._torch
        batch = self._build_batch(obs, image_variant=image_variant)
        pre, post, policy = engine["pre"], engine["post"], engine["policy"]
        with torch.no_grad():
            processed = pre(batch) if pre else batch
            chunk = policy.predict_action_chunk(processed)  # (1, H, D)
            chunk = self._unnormalize(chunk, post)
        arr = np.asarray(chunk.detach().to("cpu", dtype=torch.float32).numpy())
        arr = arr.reshape(arr.shape[-2], arr.shape[-1])  # (H, D)
        return arr[:, :7]

    def _unnormalize(self, chunk, post):
        # postprocessor で実アクション空間へ戻す。アフィン変換なので、後段で
        # 平均を取ってから戻しても等価だが、ここで戻して以降を実座標に統一する。
        if post is None:
            return chunk
        try:
            out = post({"action": chunk})
            if isinstance(out, dict):
                out = out.get("action", chunk)
            return out
        except Exception:  # noqa: BLE001 - 版差異時はそのまま実座標とみなす
            try:
                return post(chunk)
            except Exception:  # noqa: BLE001
                return chunk

    def _predict_chunk_ensemble(self, obs) -> tuple["np.ndarray", float]:
        # 全エンジン × TTA ビューでチャンクを予測し、実座標で平均して
        # (H,7) チャンクと不一致(スカラー)を返す。エンジン間で H が異なる場合は
        # 最短に合わせて切り詰める。
        np_ = self._np
        views = max(1, int(self.TTA_VIEWS))
        chunks = []
        for engine in self.engines:
            for v in range(views):
                try:
                    chunks.append(self._predict_chunk_one(obs, v, engine))
                except Exception as exc:  # noqa: BLE001 - 個々の失敗は無視
                    print(f"[MyPolicy] chunk 予測失敗 (view {v}): {exc}")
        if not chunks:
            raise RuntimeError("全てのチャンク予測に失敗しました。")
        horizon = min(c.shape[0] for c in chunks)
        stack = np_.stack([c[:horizon] for c in chunks], axis=0)  # (N, H, 7)
        mean_chunk = stack.mean(axis=0)                            # (H, 7)
        if stack.shape[0] > 1:
            # 直近ステップ(先頭)の並進 xyz の予測ばらつきを不一致とする
            disagreement = float(np_.mean(np_.std(stack[:, 0, :3], axis=0)))
        else:
            disagreement = 0.0
        return mean_chunk, disagreement

    # ------------------------------------------------------------------
    # 時間方向アンサンブル
    # ------------------------------------------------------------------
    def _aggregate(self) -> tuple["np.ndarray", float]:
        np_ = self._np
        t = self._step
        num, den = np_.zeros(7, dtype=np_.float64), 0.0
        dis_max = 0.0
        alive = []
        for start, chunk, dis in self._chunks:
            idx = t - start
            if 0 <= idx < chunk.shape[0]:
                w = float(np_.exp(-self.ENSEMBLE_DECAY * idx))
                num += w * chunk[idx].astype(np_.float64)
                den += w
                dis_max = max(dis_max, dis)
                alive.append((start, chunk, dis))
        self._chunks = alive  # 消費し切ったチャンクを掃除
        if den <= 0.0:
            return np_.zeros(7, dtype=np_.float32), dis_max
        return (num / den).astype(np_.float32), dis_max

    # ------------------------------------------------------------------
    # メイン: 観測 → アクション
    # ------------------------------------------------------------------
    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        try:
            action = self._infer(obs)
        except Exception as exc:  # noqa: BLE001 - どんな失敗でもサーバーは落とさない
            print(f"[MyPolicy] 推論に失敗しフォールバック行動を返します: {exc}")
            action = self._fallback_action()
        return self._finalize(action)

    def _infer(self, obs) -> "np.ndarray":
        np_ = self._np

        use_chunk = self._has_chunk_api and self.TEMPORAL_ENSEMBLE
        if use_chunk:
            need_new = (not self._chunks) or (self._step % max(1, self.REPLAN_INTERVAL) == 0)
            if need_new:
                chunk, dis = self._predict_chunk_ensemble(obs)
                self._chunks.append((self._step, chunk, dis))
            action, dis = self._aggregate()
            action = self._apply_uncertainty(action, dis)
            return action

        # フォールバック: full-chunk API が無い版は先頭エンジンの select_action を
        # 1 手ずつ使う（時間方向アンサンブル・多モデル平均は無効）。
        torch = self._torch
        engine = self.engines[0]
        pre, post, policy = engine["pre"], engine["post"], engine["policy"]
        batch = self._build_batch(obs, image_variant=0)
        with torch.no_grad():
            processed = pre(batch) if pre else batch
            act = policy.select_action(processed)
            act = self._unnormalize(act, post)
        act = np_.asarray(act.detach().to("cpu", dtype=torch.float32).numpy()).reshape(-1)
        if act.shape[0] < 7:
            act = np_.pad(act, (0, 7 - act.shape[0]))
        return act[:7]

    # ------------------------------------------------------------------
    # 出力整形（衝突配慮）
    # ------------------------------------------------------------------
    def _apply_uncertainty(self, action, disagreement):
        if self.UNCERTAINTY_DAMPING > 0 and disagreement > 0:
            scale = 1.0 / (1.0 + self.UNCERTAINTY_DAMPING * disagreement)
            action = action.copy()
            action[:6] *= scale  # 並進・回転のみ減速（グリッパは保持）
        return action

    def _finalize(self, action) -> np.ndarray:
        np_ = self._np
        action = np_.asarray(action, dtype=np_.float32).reshape(-1)
        if action.shape[0] < 7:
            action = np_.pad(action, (0, 7 - action.shape[0]))
        action = action[:7].astype(np_.float32)
        action = np_.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)

        # 速度スケール（並進・回転）
        if self.SPEED_SCALE != 1.0:
            action[:6] *= self.SPEED_SCALE
        # 1 ステップ並進デルタの上限（オーバーシュート抑制）
        if self.MAX_POS_DELTA < 1.0:
            action[:3] = np_.clip(action[:3], -self.MAX_POS_DELTA, self.MAX_POS_DELTA)
        # EMA 低域通過（並進・回転のみ、グリッパは応答性を保つ）
        if self.ACTION_EMA > 0 and self._prev_action is not None:
            a = self.ACTION_EMA
            action[:6] = a * self._prev_action[:6] + (1 - a) * action[:6]
        self._prev_action = action.copy()

        # グリッパのヒステリシス（チャタリング防止）
        if self.GRIPPER_MARGIN > 0:
            g = action[self.GRIPPER_IDX]
            if g > self.GRIPPER_MARGIN:
                self._gripper_state = 1.0
            elif g < -self.GRIPPER_MARGIN:
                self._gripper_state = -1.0
            action[self.GRIPPER_IDX] = self._gripper_state

        action = np_.clip(action, -1.0, 1.0).astype(np_.float32)
        self._step += 1
        return action

    def _fallback_action(self) -> "np.ndarray":
        # 推論失敗時は「直前アクションを 0 へ減衰」させ、急な動きを避ける。
        np_ = self._np
        if self._prev_action is not None:
            return (self._prev_action * 0.5).astype(np_.float32)
        return np_.zeros(7, dtype=np_.float32)


# ============================================================
# 以下は変更不可
# ============================================================


def deserialize_obs(data: bytes) -> dict[str, np.ndarray]:
    unpacked = msgpack.unpackb(data, raw=False)
    obs = {}
    for key, val in unpacked.items():
        arr = np.frombuffer(val["data"], dtype=np.dtype(val["dtype"]))
        obs[key] = arr.reshape(val["shape"]).copy()
    return obs


def serialize_action(action: np.ndarray) -> bytes:
    return msgpack.packb(
        {"data": action.astype(np.float32).tobytes()},
        use_bin_type=True,
    )


app = FastAPI(title="VLA Policy Server")
_policy: BasePolicy | None = None


def set_policy(policy: BasePolicy) -> None:
    global _policy
    _policy = policy


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
async def reset_policy(request: Request):
    body = await request.body()
    instruction = ""
    if body:
        import json
        data = json.loads(body)
        instruction = data.get("instruction", "")
    _policy.reset(instruction=instruction)
    return {"status": "ok"}


@app.post("/act")
async def act(request: Request):
    body = await request.body()
    obs = deserialize_obs(body)
    action = _policy.get_action(obs)
    return Response(
        content=serialize_action(action),
        media_type="application/x-msgpack",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    set_policy(MyPolicy())
    print(f"Policy server starting on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
