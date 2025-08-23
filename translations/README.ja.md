<div align="center">

<img src="/docs/logo.png" alt="Griffith Voice Logo" height="500" width="500">

# Griffith Voice — AI 音声クローンとダビングツール

[**English**](/README.md)｜[**简体中文**](/translations/README.zh.md)｜[**日本語**](/translations/README.ja.md)

</div>

## 🌟 概要

Griffith Voice は、リアルタイムで音声を翻訳および合成する最先端ツールです。高度な AI モデルを組み合わせて、音声の転写、翻訳、音声合成を行い、言語の壁を越えたシームレスなコミュニケーションを実現します。

主な機能：
- 🎙️ Whisper を使用したリアルタイム転写
- 🌍 GPT モデルによる多言語翻訳
- 🗣️ GPT-SoVITS を使用した高品質音声合成
- 🚀 Streamlit ベースの Web インターフェースで簡単に操作可能
- 📊 詳細なログ記録と進捗追跡

類似プロジェクトとの違い：**リアルタイム処理と高品質音声合成に特化しています。**

## 🎥 デモ

### リアルタイム翻訳とダビング

Griffith Voice の多言語リアルタイム翻訳とダビングの能力を探る。各デモでは以下を示します：
- 正確な音声転写。
- 複数の言語へのシームレスな翻訳。
- 自然な音声を使用した高品質音声合成。

<table>
<tr>
<td width="25%">

#### 英語
https://github.com/user-attachments/assets/68ccfa11-ed76-4f0e-8a33-bb8dcea31007

</td>
<td width="25%">

#### 日本語
https://github.com/user-attachments/assets/46e88ebb-a52f-4922-8391-cf1d2b2dc751

</td>
<td width="25%">

#### 韓国語
https://github.com/user-attachments/assets/60194f04-7296-40d0-8501-eaa291e94793

</td>
<td width="25%">

#### 中国語
https://github.com/user-attachments/assets/9d65ace4-115d-4e9b-b6f4-b96ae6ee6e0b

</td>
</tr>
</table>

<table>
<tr>
<td width="50%">
  
#### 英語
https://github.com/user-attachments/assets/3174b03a-fa29-4933-93e0-2fbeed349ab9

</td>
<td width="50%">

#### 日本語

https://github.com/user-attachments/assets/8e009645-5ea8-4060-abf9-b2aa40375b7b


</td>
<td width="50%">

#### 韓国語

https://github.com/user-attachments/assets/5c926032-1917-4767-8242-e1012cc33ea0

</td>
<td width="50%">

#### 中国語#### Chinese

https://github.com/user-attachments/assets/12cf313e-7535-49b6-aee2-c2e70cab877c

</td>
</tr>
</table>

### 言語サポート

**対応する入力および出力言語：**

🇺🇸 英語 | 🇯🇵 日本語 | 🇰🇷 韓国語 | 🇨🇳 中国語

**対応予定の入力および出力言語：**

🇪🇭 アラビア語 | 🇫🇷 フランス語 | 🇷🇺 ロシア語 | 🇩🇪 ドイツ語

**翻訳はすべての言語をサポートし、音声合成は選択された TTS モデルに依存します。**

## インストール

> **注意：** Python 3.10+ と FFmpeg がインストールされていることを確認してください。

1. リポジトリをクローン

```bash
git clone https://github.com/Si7li/Griffith-Voice.git
cd Griffith-Voice
```

2. 仮想環境を作成してアクティブ化

```bash
python3 -m venv env
source env/bin/activate
```

3. 依存関係をインストール

```bash
pip install -r requirements.txt
```

4. アプリケーションを起動

```bash
python streamlit_webui.py
```

## APIs
Griffith Voice は以下をサポートしています：
- 転写：Whisper
- 翻訳：GPT モデル
- 音声合成：GPT-SoVITS

## Griffith Voice を選ぶ理由

- **低 VRAM GPU に最適化**：わずか 4GB VRAM の GPU でもスムーズに動作し、幅広いユーザーにアクセス可能。
- **効率的なメモリ使用**：GPU メモリ消費を最小限に抑えながら、性能を損なわない設計。
- **リアルタイム処理**：迅速な転写、翻訳、合成を提供し、リアルタイムアプリケーションに最適。
- **デバイス間のスケーラビリティ**：高性能ワークステーションでも、一般的なセットアップでも、このツールはハードウェア能力に適応します。

これらの機能により、Griffith Voice はリアルタイムコミュニケーションとコンテンツ作成の多用途で効率的なソリューションとなります。

## 📄 ライセンス

このプロジェクトは Apache 2.0 ライセンスの下でライセンスされています。

## 📬 お問い合わせ

- GitHub で [問題](https://github.com/Si7li/Griffith-Voice/issues) または [プルリクエスト](https://github.com/Si7li/Griffith-Voice/pulls) を提出
- メール：mohamedkhalil.sahli@enicar.ucar.tn

---

<p align="center">このプロジェクトが役に立つと思ったら、ぜひ ⭐️ を付けてください！</p>
