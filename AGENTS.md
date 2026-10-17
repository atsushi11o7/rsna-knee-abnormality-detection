# AGENTS.md

## プロジェクト概要

Kaggleコンペ **RSNA Knee Abnormality Detection**(膝MRIの異常検出)用リポジトリ。
DICOM形式の膝MRI画像から異常を検出するモデルを開発する。

- コンペページ: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
- 締切: 2026-10-22 23:59 UTC(2026-10-23 08:59 JST)

## セットアップ・コマンド

- 依存追加: `uv add`。`pip install` は使わない
- 依存を変更したら `uv lock` を実行し、`uv.lock` をコミットする
- Lint: `ruff check --fix .` / Format: `ruff format .`

## コードスタイル

- Lint/format は **ruff**(`line-length=100`, `target-version=py312`, ルール: `E,F,I,B,UP`、`E501`は無視)
- コミット前に `ruff check --fix .` と `ruff format .` を通す
- マジックナンバーや非自明な仕様のみコメントを残す

## Git運用

- **base branch**: `main`
- **ブランチ命名規則**: `feature/<topic>`
- **コミットメッセージ**: タイトル1行のみ・英語の命令形・〜72文字・本文なし・`Co-Authored-By` フッターなし。例: `Add rule-based agent baseline`, `Fix deck validation`
- 1コミット = 1論理変更。`git add -A` ではなく対象ファイルを明示的に stage する
- `feature/<topic>` を切って作業し、PR経由でレビュー後 `main` へマージする。`main` への直接コミットはしない

## Pull Request

- 本文の言語: 日本語
- 本文の構成: 「変更内容」「背景・目的」

## Issue

- 本文の言語: 日本語
- 本文の構成: 種類に応じて使い分ける
  - バグ報告: 概要 / 再現手順 / 期待する挙動 / 実際の挙動
  - 機能要望: 概要 / 背景・目的 / 提案内容
  - タスク: 概要 / 背景・目的 / やること / 完了条件

## ディレクトリ方針

- `data/` … コンペ配布物・参考資料一式(gitignore対象)
- `docs/` … ローカル専用メモ(gitignore対象。例外なし)
- `submission/` … 提出物一式(Kaggle Notebook の `.ipynb` と、添付する Dataset 用の重み・コード等。gitignore対象)
- `.devcontainer/` … 開発環境定義
- `src/` … モデル学習・推論コードの置き場所(パッケージ名は検討中)

## 秘密情報の扱い

- `.env` は秘密情報。明示的な許可がない限り閲覧・編集しない。トークンやAPIキーの値はユーザー自身が設定する
- 新しい環境変数を足すときは `.env.example` を更新する
- `KAGGLE_API_TOKEN`・`GH_TOKEN` 等はログや出力に出さない
