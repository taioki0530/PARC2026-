# 参考例（examples）

| ファイル | 内容 |
|---|---|
| [smolvla_libero_spatial_lora.ipynb](smolvla_libero_spatial_lora.ipynb) | SmolVLA を LIBERO-plus Spatial で LoRA 追加学習する Google Colab ノートブック（最小構成の入門） |
| [smolvla_libero_track1_finetune.ipynb](smolvla_libero_track1_finetune.ipynb) | Track 1 の 1mm 衝突ルールに向けた広域 fine-tuning。libero_plus 全体から広くタスクを選び、アンサンブル用に複数モデルを量産する。評価は配布キット（`pipeline`/`tune.py`）の実ルール採点に委ねる |

## smolvla_libero_track1_finetune.ipynb

Track 1 の成功判定（ゴール達成 **かつ** 対象外物体の変位が全ステップで 1mm 以下）に
向けた fine-tuning ノートブック。Spatial 限定の入門ノートを土台に、次の3点で強化する。

1. **広域データ学習** — libero_plus（テクスチャ/照明などの摂動を含む）から
   object/goal も含めて広くタスクを選び、分布シフト由来の衝突を減らす
2. **アンサンブル量産** — `MODEL_TAG` / `TASK_NAME_FILTERS` / `SEED` を変えて複数回実行し、
   専門家モデルを `submission_template/model_weights/<名前>/` に並べる（MyPolicy が自動で合議）
3. **実ルール評価への委譲** — `lerobot-eval` は衝突判定をしないため本ノートでは評価せず、
   学習後にキットの `python -m pipeline` / `python tune.py` で 1mm 判定つき成功率を実測する

> 技術メモ: SmolVLA は flow-matching 損失のため「衝突ペナルティ項」を損失へ直接
> 差し込むことはできない。1mm ルールへの対応は、広域データ学習・実ルールでの
> チェックポイント選抜・推論時の衝突配慮制御（`policy_server.py` に実装済み）で行う。

### スマホ運用（Google Drive 自動保存）

「1.5 Google Drive をマウントする」セルが Drive を自動マウントし、学習後の zip を
`MyDrive/parc2026_models/<MODEL_TAG>/` へ自動コピーする。ブラウザのダウンロードに
頼らないため、Drive アプリで共有リンクを発行してチャットに貼るだけで受け渡しできる。

チェックポイントも `SAVE_FREQ`（既定: 500 step ごと、最大 STEPS）で保存されるため、
モバイルブラウザのバックグラウンド・サスペンドで途中切断しても被害が最小限に抑えられる。

> 注意: 実行自体は Google 側のサーバーで進むが、モバイルブラウザは他アプリに
> 切り替えるとタブがサスペンドされ WebSocket 接続が切れやすい（特に iOS Safari）。
> 画面を見続けられる時間が短い場合は、`STEPS` を 1500〜2000・`MAX_TASKS` を
> 10〜15 程度に下げ、一回のセッションで完走できる規模にすること。

## smolvla_libero_spatial_lora.ipynb

`lerobot/smolvla_libero_plus` を初期重みとし、LIBERO-plus Spatial の 10 タスクを
LoRA で追加学習する。学習後は LoRA を元の重みへマージし、追加学習の前後を
同一条件で比較する。

### 使い方

1. Google Colab で開き、ランタイムのタイプを GPU（T4 で足りる）に変更する
2. 上から順に実行する。所要時間は T4 で数時間程度である
3. マージ済みモデル一式（zip）と、追加学習前後の成功率の比較（CSV）が出力される

学習条件は 10 タスク × 各 5 エピソード（計 50 エピソード）、3,000 steps、
バッチサイズ 1 で、Colab で完走することを優先した最小構成である。
性能を伸ばす場合はここを出発点に、自身の環境で条件を組み直すとよい。

### 提出物にするまでの作業

出力されるのは LeRobot 形式のモデル重みであり、これ単体では提出できない。
[submission_template/](../submission_template/) の `MyPolicy` にモデルを組み込み、
ポリシーサーバーの形にする。観測と action の仕様は
[submission_template/policy_server.py](../submission_template/policy_server.py)
の docstring にある。

推論は 1 リクエストあたり 10 秒以内に収める必要がある
（[ルートの README](../README.md#タイムアウト仕様)）。

### ノートブック内の評価と、本番の採点の違い

ノートブック内の評価は学習の効果を手早く確認するためのもので、採点とは条件が異なる。
出てくる成功率は本番スコアの目安にはならない。

| 項目 | ノートブック | 本番の採点 |
|---|---|---|
| 評価タスク | LIBERO-plus Spatial の 10 タスク | Track 1（`compe/t1/` のタスクセット） |
| 実行方法 | LeRobot の `lerobot-eval` | `python -m pipeline` + 提出したポリシーサーバー |
| 観測の解像度 | 256×256 | 128×128 |
| 1 タスクあたりの試行数 | 3（`EVAL_EPISODES_PER_TASK` で変更可） | 非公開（配布キットの既定は 20） |

試行数が 3 のままだと 1 エピソードの成否で成功率が約 33 ポイント動くため、
追加学習の前後を比べる場合は `EVAL_EPISODES_PER_TASK` を増やすこと。

### 実行環境

ノートブックの環境構築は Colab 向けで、[setup.sh](../setup.sh) とは独立している。
依存パッケージのバージョンが一致しない箇所があるため、評価と提出前チェックは
リポジトリ側の環境（`setup.sh` + `env.sh`）で行うこと。

ノートブックが利用する第三者製ソフトウェア・モデル・データセットのライセンスは、
各配布元の表記を参照すること。
