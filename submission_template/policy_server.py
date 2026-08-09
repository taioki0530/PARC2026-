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
    """SmolVLA（LeRobot）を用いたポリシー実装。

    `examples/smolvla_libero_spatial_lora.ipynb` で学習・マージした
    LeRobot 形式の SmolVLA 重み一式（`config.json` / `model.safetensors` /
    `policy_preprocessor*.{json,safetensors}` / `policy_postprocessor*.{json,safetensors}`）を
    このファイルと同じ階層の `model_weights/` に配置して使用する。

        submission.zip
        ├── policy_server.py
        ├── requirements.txt
        └── model_weights/        # ← save_pretrained() の出力一式をそのまま置く
            ├── config.json
            ├── model.safetensors
            ├── policy_preprocessor.json
            ├── policy_preprocessor.safetensors
            ├── policy_postprocessor.json
            └── policy_postprocessor.safetensors

    採点環境は外部通信を遮断するため、重みは必ず同梱すること
    （`from_pretrained` がネットワークへ取りに行かないよう、リポジトリ ID ではなく
    ローカルパスを渡している）。

    観測・アクションの仕様は LeRobot の LIBERO 評価ラッパに合わせている:
      - agentview → observation.images.image / eye_in_hand → observation.images.image2
      - observation.state(8) = eef_pos(3) + quat→axis-angle(3) + gripper_qpos(2)
      - action(7) = [dx, dy, dz, droll, dpitch, dyaw, gripper]（control_mode=relative, [-1, 1]）
    action chunking は SmolVLA 内部のキューで処理され、`reset()` でクリアする。
    """

    # ---- 設定（必要に応じて編集する）---------------------------------------
    # LeRobot 形式の重みを置いたディレクトリ（policy_server.py からの相対パス）
    MODEL_DIR = "model_weights"
    # LIBERO の OffScreenRenderEnv は画像を上下反転で返すため、既定で反転して
    # 学習時（LeRobot データセット）と向きを揃える。ローカル検証で見え方が
    # おかしい場合はここを False にして確認すること。
    FLIP_IMAGES = True
    # 生の観測キー → モデル入力特徴キーの対応
    CAMERA_KEYS = {
        "agentview_image": "observation.images.image",
        "robot0_eye_in_hand_image": "observation.images.image2",
    }

    def __init__(self):
        import os

        import torch

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.instruction = ""

        model_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), self.MODEL_DIR
        )
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"モデルディレクトリが見つかりません: {model_dir}. "
                "LeRobot 形式の SmolVLA 重み一式を model_weights/ に配置してください。"
            )

        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        # ローカルパスからロード（採点環境は外部通信を遮断するため）
        self.policy = SmolVLAPolicy.from_pretrained(model_dir)
        self.policy.to(self.device)
        self.policy.eval()

        # LeRobot v0.6 系ではモデルとは別に前後処理パイプライン
        # （正規化・トークナイズ等）を保存・ロードする。存在すればそちらを使い、
        # 無い/API 差異でロードできない場合は policy 内部の正規化に委ねる。
        self.preprocessor = None
        self.postprocessor = None
        try:
            from lerobot.processor import make_pre_post_processors

            self.preprocessor, self.postprocessor = make_pre_post_processors(
                self.policy.config,
                pretrained_path=model_dir,
            )
        except Exception as exc:  # noqa: BLE001 - 版差異は握りつぶし内部正規化へフォールバック
            print(
                "[MyPolicy] 前後処理パイプラインをロードできませんでした "
                f"({exc}). policy 内部の正規化にフォールバックします。"
            )

    def reset(self, instruction: str = "") -> None:
        # エピソード開始時に呼ばれる。言語指示を保持し、action chunk の
        # 内部キャッシュ（キュー）をクリアする。
        self.instruction = instruction or ""
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    @staticmethod
    def _quat_to_axis_angle(quat) -> "np.ndarray":
        # robosuite の quaternion 規約は [x, y, z, w]。
        q = np.asarray(quat, dtype=np.float64).reshape(4)
        norm = np.linalg.norm(q)
        if norm < 1e-8:
            return np.zeros(3, dtype=np.float64)
        q = q / norm
        # w を正に揃えて回転角を [-pi, pi] に収める
        if q[3] < 0:
            q = -q
        angle = 2.0 * np.arccos(np.clip(q[3], -1.0, 1.0))
        sin_half = np.sqrt(max(0.0, 1.0 - q[3] * q[3]))
        if sin_half < 1e-8:
            return np.zeros(3, dtype=np.float64)
        axis = q[:3] / sin_half
        return (axis * angle).astype(np.float64)

    def _build_batch(self, obs: dict[str, np.ndarray]) -> dict:
        torch = self._torch
        batch: dict = {}

        # 画像: HWC uint8 → CHW float32[0,1]、バッチ次元付与
        for raw_key, feat_key in self.CAMERA_KEYS.items():
            img = obs[raw_key]
            if self.FLIP_IMAGES:
                img = img[::-1]
            img = np.ascontiguousarray(img)
            tensor = torch.from_numpy(img).to(torch.float32) / 255.0
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            batch[feat_key] = tensor.to(self.device)

        # 状態(8) = eef_pos(3) + axis-angle(3) + gripper_qpos(2)
        eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)[:3]
        axis_angle = self._quat_to_axis_angle(obs["robot0_eef_quat"]).astype(np.float32)
        gripper = np.asarray(
            obs["robot0_gripper_qpos"], dtype=np.float32
        ).reshape(-1)[:2]
        state = np.concatenate([eef_pos, axis_angle, gripper]).astype(np.float32)
        batch["observation.state"] = (
            torch.from_numpy(state).unsqueeze(0).to(self.device)
        )

        # 言語指示（SmolVLA は batch["task"] を参照する）
        batch["task"] = [self.instruction]
        return batch

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        torch = self._torch
        batch = self._build_batch(obs)

        with torch.no_grad():
            if self.preprocessor is not None:
                processed = self.preprocessor(batch)
                action = self.policy.select_action(processed)
                if self.postprocessor is not None:
                    action = self.postprocessor(action)
            else:
                # フォールバック: policy 内部で正規化・逆正規化が行われる版
                action = self.policy.select_action(batch)

        action = np.asarray(
            action.detach().to("cpu", dtype=torch.float32).numpy()
        ).reshape(-1)

        # 形状を (7,) に整える
        if action.shape[0] < 7:
            action = np.pad(action, (0, 7 - action.shape[0]))
        action = action[:7]

        # 数値健全性: NaN/Inf を除去し、行動範囲 [-1, 1] にクリップ
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        action = np.clip(action, -1.0, 1.0)
        return action.astype(np.float32)


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
