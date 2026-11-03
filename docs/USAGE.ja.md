# HF モデル一括ダウンロード (GUI 付き) 使い方

固定リビジョンの Hugging Face モデルをまとめて取得するためのツール。
ブラウザの画面で、保存先の指定、モデル選択、一時トークンの投入、ダウンロード、検証までできる。
GUI を使わず CLI だけでも動く。

## 1. 起動

必要なもの: Python 3.10 以上、PyPI と huggingface.co へ出られるネットワーク。

OS ごとに起動スクリプトが 1 つずつある。どれも初回だけ `.venv` を作って `requirements.txt` を入れ、サーバーを起動し、ブラウザで GUI を自動的に開く。

| OS | 起動スクリプト | 起動方法 |
|---|---|---|
| Windows | `start_gui_windows.bat` | エクスプローラーでダブルクリック(またはコマンドプロンプトから実行) |
| macOS | `start_gui_mac.command` | Finder でダブルクリック(またはターミナルから実行) |
| Linux | `start_gui_linux.sh` | `./start_gui_linux.sh` |

macOS で zip を展開した直後は、ダブルクリックが「開けません」と止められることがある。その場合は右クリックから「開く」を選ぶか、ターミナルで一度だけ次を実行する。

```bash
xattr -dr com.apple.quarantine hf_model_downloader_gui_bundle
chmod +x hf_model_downloader_gui_bundle/start_gui_mac.command
```

Python 3.10 以上が見つからないときは、スクリプトが導入方法を表示して止まる(macOS 標準の python3 は古いことが多いので `brew install python@3.12` か python.org から入れる)。`.venv` を別の OS からコピーして持ち込んだ場合は、使えないと判定して自動的に作り直す。

起動するとターミナルに次のような URL が出て、ブラウザが開く。

```
open : http://127.0.0.1:8765/?k=<毎回変わるキー>
```

この URL のキーがないと画面も API も使えない。ターミナルに出た URL をそのまま開くこと。

### GPU サーバーなどリモートマシンで動かす場合

GUI はそのマシンの 127.0.0.1 でしか待ち受けない(外部公開する設定は意図的に無い)。手元の PC から SSH ポート転送で見る。

```bash
# 遠隔側 (デスクトップの無い Linux では --no-browser を付けなくてもブラウザは開かない)
./start_gui_linux.sh --no-browser

# 手元側(別ターミナル)
ssh -L 8765:127.0.0.1:8765 user@gpu-host
# 遠隔側のターミナルに出た http://127.0.0.1:8765/?k=... を手元のブラウザで開く
```

## 2. 画面の使い方

1. 保存先: 絶対パスを入力するか「参照...」でフォルダを選ぶ(その場でフォルダ作成も可)。空き容量と書込可否が出る。
2. モデル: グループ(core3 / core4 / nv_edge など)を選ぶか、チェックボックスで個別選択。一覧が長いので、名前・役割の検索、形式(BF16 / FP8 / NVFP4 / GGUF など)、サイズ帯(10GB未満〜500GB超)で絞り込める。絞り込みは表示だけを変え、チェックは保持される。「表示中を全選択 / 全解除」で絞り込んだ分だけまとめて操作できる。例えば形式を NVFP4、サイズ帯を 50-150GB にして全選択すれば、その範囲だけが選ばれる。
   - GGUF のファイルが複数あるリポジトリ(Swift の IQ2_XS / IQ3_S、Bolt 74B GGUF の本体 / mmproj など)は、名前の下に「取得するファイル」のチェックが出て、要るファイルだけ選べる。README やライセンスなど GGUF 以外のファイルは常に取得する。サイズ欄と合計は選んだ分で計算され、全部の場合との差も表示される。分割された GGUF(`-00001-of-00003.gguf` など)は 1 つのまとまりとして扱う。後から別のファイルを追加で選んで実行すると、取得済みのファイルはそのまま残り、足りない分だけ取得する。
3. トークン(任意): ほとんどのモデルは公開・認可不要なので、無くても取得できる。レート制限や接続断への余裕が欲しいとき、または承認制モデルを取るときに入れる。
4. ダウンロード: 「事前確認」は何も作らず何も取らない。「ダウンロード開始」は確認ダイアログの後に開始する。「停止」で止めても、同じ保存先でもう一度開始すれば続きから再開する。
5. 検証: ダウンロード後に実行する。

## 3. トークンの扱い(一時 ReadKey)

- 入力した `hf_...` は、まず Hugging Face に問い合わせて有効性を確認する。無効なら保存しない。
- 保存先はこのサーバープロセスのメモリだけ。ファイルにも環境変数にも書かない。ダウンロードや検証の作業プロセスには標準入力で渡す。
- 画面・ログ・API 応答には末尾4文字しか出ない。ログ出力は全てマスクされる。
- 消えるタイミング: 「消去」ボタン、設定した有効時間(1時間 / 4時間 / 24時間)、ジョブ完了後の自動消去(既定オン)、「終了」ボタンまたは Ctrl+C。
- 書込権限を持つトークンを入れると警告が出る。ダウンロードには Read 権限のみ(または fine-grained の read)で足りる。
- GUI でトークンをセットしていないときは、huggingface_hub が自分でトークンを探して使う。順番は環境変数 `HF_TOKEN`、次に `hf auth login`(または `huggingface_hub.login()`)で保存されたログイン済みトークン。どちらが使われるかは画面とログに出る(値は出ない)。GUI でセットしたトークンはこれらより優先される。
- 「消去」ボタンが消すのは GUI でセットしたトークンだけ。ログイン済みトークンを使わせたくない場合は、ターミナルで `hf auth logout` を実行する(環境変数 `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` でも自動送信を止められる)。
- 承認制モデルを選んでトークンが一つも無いと、GUI は開始前に止める。ログイン済みトークンがあれば止めない。

## 4. 検証の種類

| ボタン | 何を確認するか | 向いている場面 |
|---|---|---|
| HF公式ハッシュと照合(全ファイル) | Hugging Face が公開しているハッシュ(LFS は SHA-256、通常ファイルは git blob SHA-1)と全ファイルを突き合わせる | ダウンロード直後。ダウンロード中の破損や取りこぼしを検出する |
| サイズだけ高速確認 | 全ファイルの存在とサイズのみ | 途中で止めたあとの足りないファイル探し。破損は検出できない |
| sha256sums.txt を作成 | 手元のコピーから控えを作る(`.cache` は除外) | 別マシンへ移送する前 |
| sha256sums.txt で検証 | 控えと突き合わせ、欠落・不一致・余分なファイルを報告 | 移送先で |

注意: sha256sums.txt は手元のコピーから作るので、ダウンロード時点で壊れていたものは「正しい」と記録される。ダウンロードの正しさは「HF公式ハッシュと照合」で確認する。

## 5. CLI で使う場合

```bash
# 計画と空き容量の確認(何も作らない)
./download_models.sh --dry-run --group core4 --dest /Volumes/MODELS/hf_models

# ダウンロード(トークン無し)
./download_models.sh --group core4 --dest /Volumes/MODELS/hf_models

# トークンは標準入力で渡せる(環境変数 HF_TOKEN でも可)
read -rs HF_TOK && printf '%s\n' "$HF_TOK" | .venv/bin/python download_models.py --token-stdin --group core4 --dest /Volumes/MODELS/hf_models

# GGUF のファイルが複数あるリポジトリから、一部だけ取得する(値は model_manifest.json の gguf_variants の label か id)
./download_models.sh --only Kujira/Swift-1.5-Qwen3.8-27B-CHiPS-Uncensored-GSQ-RCO-MTP-GGUF \
  --variant Kujira/Swift-1.5-Qwen3.8-27B-CHiPS-Uncensored-GSQ-RCO-MTP-GGUF=IQ3_S-mtp --dest /Volumes/MODELS/hf_models

# 検証
.venv/bin/python verify_hf_hashes.py --dest /Volumes/MODELS/hf_models --group core4
./sha256_generate.sh /Volumes/MODELS/hf_models
./sha256_verify.sh   /Volumes/MODELS/hf_models
```

`download_models.py` の主なオプション: `--only <repo_id>`(複数可)、`--variant <repo_id>=<ファイル>`(複数可)、`--max-workers`、`--retries`、`--retry-wait`、`--high-performance`、`--space-factor`。

`verify_hf_hashes.py` も `--variant` を受け付ける。付けない場合、ファイルが複数ある GGUF リポジトリでは、手元に 1 つでもファイルがあるものだけを照合し、手元に無いものは「スキップ」と表示する(エラーにはしない)。

## 6. グループと容量

サイズは固定リビジョンのファイル合計を HF API で実測した値(2026-11-03)。

| グループ | 内容 | 合計 |
|---|---|---|
| core3 | Qwen3.8-27B + DeepSeek V4.1 Flash + GLM-5.3 Flash | 約 655 GB |
| core4 | core3 + Qwen3.8 Flash Next | 約 788 GB |
| no_coder | core4 + Gemma 4 31B | 約 851 GB |
| all | 元の 6 モデル(no_coder + Qwen3-Coder 480B) | 約 1.81 TB |
| new_small | 追加分の小型(Nemotron 3.5 Lightning + Super VL) | 約 97 GB |
| new_mid | 追加分の中型(MiMo V2.6 Flash、MiniMax M3.1 Flash、Reflection Beam) | 約 724 GB |
| new_large | 追加分の大型(下表の 6 モデル) | 約 3.15 TB |
| osi_open | 追加分のうち MIT / Apache-2.0 のみ(MiMo Pro・Flash、Ling 3.1、Beam) | 約 1.62 TB |
| nv_edge | NVIDIA 小メモリ向け(GGUF 3 種と、24GB に収まる Bolt 74B の safetensors / GGUF) | 約 61 GB |
| nv_nvfp4 | NVIDIA 自身の Nemotron 系の NVFP4 全部(Labs 系を含む、Blackwell 向け) | 約 1.05 TB |
| nv_fp8 | NVIDIA Nemotron 系の FP8 全部(Hopper / Ada / Blackwell 向け) | 約 264 GB |
| nv_bf16 | NVIDIA Nemotron 系の BF16 全部(どの GPU でも可、Ultra だけで約 1.1 TB) | 約 2.09 TB |
| nvq_current | NVIDIA が NVFP4 化した他社モデル、2026 年リリース分(28 件) | 約 10.3 TB |
| nvq_legacy | 同、2025 年世代(19 件) | 約 2.37 TB |
| community | コミュニティ改変の GGUF(依頼で追加した 2 件) | 約 29 GB |

### 追加モデル(2026-11-03 に調査して追加)

HF 上で実在・非 private を確認し、リビジョンを固定してサイズを実測した。形式は Blackwell 世代の GPU 向けに NVFP4 があればそれを選んだ(無いものは FP8 / MXFP8)。

| モデル | サイズ | 形式 | ライセンス(HF 表記) |
|---|---|---|---|
| nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 | 21.6 GB | NVFP4 | openmdw-1.1 |
| nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 | 75.1 GB | NVFP4 | openmdw-1.1 |
| XiaomiMiMo/MiMo-V2.6-Flash-RL | 177.8 GB | FP8 | MIT |
| MiniMaxAI/MiniMax-M3.1-Flash | 252.4 GB | MXFP8 | MiniMax Community(既定で非商用) |
| reflection/Beam-1.0-501B-A23B-NVFP4 | 293.7 GB | NVFP4 | Apache-2.0 |
| stepfun-ai/Step-5-Preview-NVFP4 | 394.2 GB | NVFP4 | StepFun Model License 1.0 |
| nvidia/GLM-5.3-NVFP4 | 464.2 GB | NVFP4 | NVIDIA Open Model License(元の GLM-5.3 にも独自ライセンス) |
| meta-models/Muse-Spark-Open-816B-A42B-NVFP4 | 533.0 GB | NVFP4 | Muse Model License(承認制) |
| XiaomiMiMo/MiMo-V2.6-Pro-MOPD | 573.5 GB | FP8 | MIT |
| inclusionAI/Ling-3.1-flash-1019 | 573.8 GB | FP8 | MIT |
| mistralai/Mistral-Large-4-1T-A52B-NVFP4 | 610.1 GB | NVFP4 | Mistral Ecosystem Open-Weight(MaaS 事業は別契約) |

- Muse Spark Open は承認制(gated)。HF のモデルページでライセンスに同意し承認を受けたアカウントのトークンが必要。GUI はトークン未設定のままこのモデルを選んで開始しようとすると止める。
- 追加を見送った候補は `model_manifest.json` の `candidates_checked_not_added` に理由付きで残してある(Kimi K3 は 1.61 TB で、Kimi K3.1 Pro の重みが 11/4 までに公開予定のため待ち、など)。
- どのモデルが強いかの順位は Artificial Analysis などの第三者集計の記事に基づく。ベンダー自己申告を含むので、採用前に自分の用途で評価すること。

### NVIDIA 公式モデル(Nemotron 系、軽量から重量級まで)

NVIDIA 自身が出しているチャット / 推論用の LLM と VLM を、軽量から重量級まで入れた。Blackwell 以外の GPU でも使えるよう、同じモデルを形式違い(NVFP4 / FP8 / BF16 / GGUF)で別項目にしてある。GUI では形式の絞り込みで選ぶ。

| モデル | 規模 | 形式とサイズ(GB) | ライセンス(HF 表記) |
|---|---|---|---|
| Nemotron 3.5 Bolt E4B(Gemma-4-E4B-it ベース、エッジ向け) | 8B(実効4B) | GGUF 6.2 / BF16 16.1 | OpenMDW-1.1 |
| Nemotron 3 Nano 4B | 4B | GGUF 2.8 / FP8 5.3 / BF16 8.0 | NVIDIA Nemotron Open Model |
| Nemotron Labs Dory 1.0 4B(ループ型 LM、研究) | 4B | GGUF 3.7 / BF16 8.0 | OpenMDW-1.1 |
| Nemotron Labs Diffusion 3B / 14B(拡散 LM、研究) | 3B / 14B | BF16 7.8 / 27.2 | NVIDIA Nemotron Open Model |
| Nemotron Nano 9B v2 | 9B | NVFP4 7.9 / FP8 10.3 / BF16 17.8 | NVIDIA Open Model |
| Nemotron Nano 12B v2 VL | 12B | NVFP4 10.6 / BF16 26.4 | NVIDIA Open Model |
| Nemotron 3 Nano 30B-A3B | 30B / 3B active | NVFP4 19.4 / FP8 32.7 / BF16 63.2 | NVIDIA Nemotron Open Model |
| Nemotron 3.5 Lightning 30B-A3B | 30B / 3B active | NVFP4 21.6 / BF16 65.8 | OpenMDW-1.1 |
| Nemotron Cascade 2 30B-A3B(数学・コード推論) | 30B / 3B active | BF16 63.2 | NVIDIA Open Model |
| Nemotron 3 Nano Omni 30B-A3B(映像・音声・画像・テキスト入力) | 31B / 3B active | NVFP4 22.4 / FP8 35.2 / BF16 66.1 | NVIDIA Open Model Agreement |
| Nemotron 3.5 Bolt 74B-A9B 24GB(2bit QAT、24GB に収まる VLM) | 74B / 9B active | safetensors 24.1 / GGUF 23.8 | OpenMDW-1.1 |
| Llama-3.3-Nemotron-Super 49B v1.5(2025年版) | 49B | NVFP4 31.1 / FP8 52.0 / BF16 99.8 | NVIDIA Open Model(Llama 3.3 ベース) |
| Nemotron 3 Super 120B-A12B | 120B / 12B active | NVFP4 80.4 / FP8 128.4 / BF16 247.2 | NVIDIA Nemotron Open Model |
| Nemotron 3.5 Super VL 120B-A12B | 120B / 12B active | NVFP4 75.1 / BF16 248.9 | OpenMDW-1.1 |
| Nemotron 3 Ultra 550B-A55B | 550B / 55B active | NVFP4 352.4 / BF16 1121.1 | OpenMDW-1.1 |
| Nemotron Labs 3 Puzzle 75B-A9B(Super 120B を圧縮した派生) | 75B / 9B active | NVFP4 53.5 | OpenMDW-1.1 |
| Nemotron Labs 3 Competitive Coding 550B-A55B(競技プログラミング特化、Ultra ベース) | 550B / 55B active | NVFP4 352.4 | OpenMDW-1.1 |
| Nemotron Labs 3 Elastic 30B-A3B(30B / 23B / 12B を 1 つに内包) | 30B / 3B active | NVFP4 19.4 | NVIDIA Open Model |
| Nemotron Labs Dory 1.0 4B | 4B | NVFP4 3.2 | OpenMDW-1.1 |

- GGUF のリポジトリは量子化 1 種だけが入っているので、そのまま全部取得する形で足りる。
- Gemma-4-E4B-it ベースの Bolt E4B や、Llama 3.3 ベースの 49B のように、元モデルのライセンス条件も確認が必要なものがある。
- 入れなかったもの: 埋め込み・リランカー・音声認識・文書パース・コンテンツ安全・Cosmos・Alpamayo などの非 LLM、Base チェックポイント、投機的デコード用ドラフトモデル、他社モデルを NVIDIA が NVFP4 化したもの(Qwen3.6-35B-A3B、Gemma-4-31B-IT、Kimi K2 系など。一部は既に別項目で入っている)。理由は `model_manifest.json` の `candidates_checked_not_added` に残してある。
- どの形式が動くかは GPU 次第。NVFP4 は Blackwell(B200 / B300 / GB200 / GB300 / RTX 50 系)向け、FP8 は Hopper / Ada 以降、BF16 はどの GPU でも可、GGUF は llama.cpp 向け。カードに最小 GPU が書かれているもの(Nano Omni、Bolt 74B)は、その内容を項目の注記(マウスを載せると出る)に入れてある。他のモデルは各カードで確認すること。

### NVIDIA が NVFP4 化した他社モデル(Blackwell 向け)

NVIDIA が Hugging Face で公開している NVFP4 版のうち、他社のモデルを量子化したものを別グループにした。NVIDIA のモデルではないので、元モデルのライセンスも別途適用される。GUI では形式を NVFP4 に絞り、検索欄に元モデル名を入れると探せる。

- `nvq_current`: 2026 年に公開された 28 件、約 10.3 TB。Kimi 系が各約 0.6 TB、Kimi K3 が約 1.6 TB、Qwen3.8-2.4T が約 1.4 TB、DeepSeek V4 Pro 系が各約 0.9 TB と大きいので、丸ごと取得せず絞って選ぶこと。
- `nvq_legacy`: 2025 年世代の 19 件、約 2.37 TB。
- 公開日で振り分けているため、Qwen3-Coder-480B のように元モデルが 2025 年で NVFP4 版が 2026 年のものは `nvq_current` に入っている。
- 「Local」と付く 2 件(Qwen3.8-Flash-Next-FP4-Local、GLM-5.3-Flash-FP4-Local)は、NVFP4 のスケール係数に新しい圧縮形式を使っており、本家 vLLM がまだ対応していない。カードに書かれた NVIDIA 提供のコンテナを使う必要がある。Qwen3.8-Flash-Next 側のカードには、RTX PRO 6000 Blackwell と DGX Spark で検証したと書かれている。
- Llama-4-Scout は承認制(gated)。HF でライセンスに同意したアカウントのトークンが必要で、GUI はトークン未設定だと止める。
- 取り込まなかったもの: 新しい版に置き換わったもの(DeepSeek-V4.1-Flash 9 月版、Qwen3.5-397B の初版、DeepSeek-R1 系の旧版、V3-0324 / V3.1、Qwen3-235B の初版)、量子コンピュータの校正用の Ising-Calibration、ベンチマーク用・投機デコード用ドラフト、画像・動画・埋め込み・音声のモデル。理由は `model_manifest.json` の `candidates_checked_not_added` にある。

#### nvq_current(2026 年)

| 元モデル | サイズ(GB) | ライセンス(HF 表記) |
|---|---|---|
| deepseek-ai/DeepSeek-V4-Flash | 168.3 | mit |
| deepseek-ai/DeepSeek-V4-Flash-0731 | 175.6 | mit |
| deepseek-ai/DeepSeek-V4-Pro | 913.1 | mit |
| deepseek-ai/DeepSeek-V4-Pro-0813 | 941.1 | mit |
| google/diffusiongemma-26B-A4B-it | 18.9 | apache-license-2.0 |
| google/gemma-4-26B-A4B-it | 18.8 | apache-license-2.0 |
| google/gemma-4-31B-it | 32.7 | apache-license-2.0 |
| meta-models/Muse-Glimmer-30B | 24.7 | apache-2.0 |
| MiniMaxAI/MiniMax-M2.5 | 139.9 | nvidia-open-model-license |
| MiniMaxAI/MiniMax-M2.7 | 139.9 | nvidia-software-and-model-evaluation-license |
| MiniMaxAI/MiniMax-M3 | 250.1 | minimax-community-license |
| mistralai/Mistral-Medium-3.5-128B | 95.3 | nvidia-open-model-license |
| moonshotai/Kimi-K2.5 | 590.9 | nvidia-open-model-license |
| moonshotai/Kimi-K2.6 | 595.2 | nvidia-open-model-license |
| moonshotai/Kimi-K2.7-Code | 595.2 | nvidia-open-model-license |
| moonshotai/Kimi-K3 | 1610.0 | nvidia-open-model-license |
| Qwen/Qwen3-Coder-480B-A35B-Instruct | 272.9 | apache-2.0 |
| Qwen/Qwen3.5-122B-A10B | 83.5 | apache-2.0 |
| Qwen/Qwen3.5-397B-A17B | 243.7 | apache-2.0 |
| Qwen/Qwen3.6-27B | 21.9 | apache-2.0 |
| Qwen/Qwen3.6-35B-A3B | 23.5 | apache-2.0 |
| Qwen/Qwen3.8-2.4T-A95B | 1444.6 | nvidia-open-model-license |
| Qwen/Qwen3.8-Flash-Next | 102.3 | nvidia-open-model-agreement |
| zai-org/GLM-4.7 | 230.0 | mit |
| zai-org/GLM-5 | 480.8 | mit |
| zai-org/GLM-5.1 | 465.9 | mit |
| zai-org/GLM-5.2 | 464.9 | mit |
| zai-org/GLM-5.3-Flash | 177.5 | nvidia-open-model-agreement |

#### nvq_legacy(2025 年世代)

| 元モデル | サイズ(GB) | ライセンス(HF 表記) |
|---|---|---|
| deepseek-ai/DeepSeek-R1-0528 | 413.3 | mit |
| deepseek-ai/DeepSeek-V3.2 | 415.1 | mit |
| meta-llama/Llama-3.1-405B-Instruct | 234.3 | llama3.1 |
| meta-llama/Llama-3.1-8B-Instruct | 6.0 | nvidia-open-model-license |
| meta-llama/Llama-3.3-70B-Instruct | 42.7 | llama3.3 |
| meta-llama/Llama-4-Scout-17B-16E-Instruct(要トークン) | 65.4 | nvidia-open-model-license |
| microsoft/Phi-4-multimodal-instruct | 9.0 | nvidia-open-model-license |
| microsoft/Phi-4-reasoning-plus | 9.7 | mit |
| moonshotai/Kimi-K2-Thinking | 594.3 | nvidia-open-model-license |
| Qwen/Qwen2.5-VL-7B-Instruct | 7.2 | nvidia-open-model-license |
| Qwen/Qwen3-14B | 10.6 | apache-2.0 |
| Qwen/Qwen3-235B-A22B-Instruct-2507 | 139.2 | apache-2.0 |
| Qwen/Qwen3-235B-A22B-Thinking-2507 | 139.2 | apache-2.0 |
| Qwen/Qwen3-30B-A3B | 18.1 | apache-2.0 |
| Qwen/Qwen3-32B | 20.7 | apache-2.0 |
| Qwen/Qwen3-8B | 6.4 | apache-2.0 |
| Qwen/Qwen3-Next-80B-A3B-Instruct | 50.8 | apache-2.0 |
| Qwen/Qwen3-Next-80B-A3B-Thinking | 50.8 | apache-2.0 |
| Qwen/Qwen3-VL-235B-A22B-Instruct | 135.3 | see base model |

### コミュニティ改変の GGUF(グループ `community`)

依頼で追加した 2 件。どちらも公式リリースではなく、第三者が拒否応答(refusal)を減らす編集をした非公式ビルドで、llama.cpp 向けの GGUF。元モデルの提供元による検証は受けていない。

| モデル | サイズ(GB) | 内容 | ライセンス(HF 表記) |
|---|---|---|---|
| Kujira/Underdog-Saluki-27B-1.0-MTP-Abliterated-GGUF | 8.35 | IQ2-mix 1 ファイル。MTP ヘッド付き | Apache-2.0 |
| Kujira/Swift-1.5-Qwen3.8-27B-CHiPS-Uncensored-GSQ-RCO-MTP-GGUF | 20.9 | IQ2_XS(8.77)と IQ3_S(12.12)の 2 ファイル。MTP ヘッド付き | swift-open-license-1.0(リポジトリに Apache-2.0 の文書も同梱) |

- カードによると、どちらも 4 テンソルを書き換えただけで、学習はしていない。手法は非公開。MTP は llama.cpp の `--spec-type draft-mtp`(自己投機的デコード)で使う。
- Swift 側の評価結果(拒否応答 93.3% から 2.9%、GSM8K・MMLU-Pro はほぼ維持)は IQ2_XS のみ。IQ3_S は同じ編集だが未評価と書かれている。カードの数字は作者の自己申告。
- Swift 側は 2026-11-02 作成で、確認時点でダウンロード・いいねとも 0 件。
- 拒否応答が減っているので、顧客向け・本番用途では自前のガードレールを別に用意すること。

初版の manifest では GLM-5.3-Flash が約 409 GB、Gemma 4 31B が約 176 GB と書かれていたが、実測は約 204 GB と約 63 GB だった。

## 7. 初版からの変更点

- `sha256_generate.sh` が各モデル直下の `.cache`(再開用メタデータ)をハッシュ対象に含めていた問題を修正(Python 実装に統一し、あらゆる深さの `.cache` を除外)。
- 1 モデルの失敗で全体が止まる問題を修正。通信エラーは指数バックオフで再試行し、失敗しても次のモデルへ進む。認証・存在しないリポジトリ・ディスク満杯などリトライしても直らない失敗は即座に次へ進む。終了コードは失敗があれば 1。
- `--dry-run` が保存先ディレクトリを作る副作用を削除。
- `download_models.sh` が `cd` して相対パスの `--dest` がずれる問題を修正。
- 空き容量が残りのダウンロード量より小さい場合は開始せず終了する(従来は警告のみ)。取得済みの分は差し引いて判定する。
- manifest のサイズを実測値に修正。
- HF 公式ハッシュとの照合を追加。
- `.sh` の実行権限を付与(zip 内で 0755)。

## 8. 制約と注意

- 進捗(バイト数と速度)は保存先のディスク使用量から推定している。ダウンロード全体の完了判定は作業プロセスの終了コードで行う。
- ジョブは同時に 1 つだけ。
- `download_models.sh` / `sha256_*.sh` は bash 用(macOS / Linux)。Windows では `start_gui_windows.bat` から GUI を使うか、`python download_models.py` を直接実行する。
- ブラウザが自動で開かない環境(デスクトップの無い Linux、SSH 経由など)では、ターミナルに出た URL を手で開く。起動時に `--no-browser` を付けると自動起動を止められる。
- 起動オプション(全 OS 共通): `--port 9000`(既定 8765、使用中なら次の空きポート)、`--default-dest <パス>`(保存先の初期値)、`--no-browser`。
- 停止すると作業プロセスを強制終了する。途中のファイルは `.cache` に残り、再開時に続きから取得される。
- `Qwen3.8-Flash-Next-NVFP4` のライセンスは HF 上のタグが `other`(NVIDIA 独自)。利用前に条文を確認すること。
- テスト: `python -m unittest discover -s tests -v`(極小の公開モデル約 45 MB を一時フォルダに取得する。インターネット接続が必要)。
