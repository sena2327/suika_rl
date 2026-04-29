---
theme: seriph
title: Suika RL 実験報告
info: |
  スイカゲーム強化学習の実験まとめ
class: text-center
drawings:
  persist: false
transition: slide-left
comark: true
---

# スイカゲーム強化学習 実験報告

Sena Goto  
2026-04-29

---
layout: two-cols
layoutClass: gap-8 items-center
---

# スイカゲームの紹介

- 同じ種類のフルーツをぶつけて1段階大きくする物理パズル
- 落下位置 $x$ を連続制御して、最終スコア最大化を目指す
- 本実験では 既存のChrome-seluenium環境をnode.js環境に変更し、PPOで学習

参照環境: [edwhu/suika_rl](https://github.com/edwhu/suika_rl)



::right::

<div class="h-full flex items-center justify-center">
  <img src="/assets/suika.jpeg" alt="suika game" class="w-full max-w-sm rounded shadow" />
</div>

---

# 学習手法: PPO（概要）

目的関数（clipped surrogate）

$$
L^{CLIP}(\theta)=\hat{\mathbb{E}}_t\left[\min\left(r_t(\theta)\hat A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t\right)\right]
$$

$$
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

Advantage（GAE）

$$
\hat A_t=\sum_{l=0}^{\infty}(\gamma\lambda)^l\,\delta_{t+l},
\quad
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
$$

更新時は以下を合成
- Policy loss（上式）
- Value loss（価値関数 \(V(s)\) の予測誤差を減らす）
- Entropy loss（方策のエントロピーを保ち、探索を促す）

---

# 実験基本設定

- 環境: `SuikaEnvNode-v0`（Node.js 実装）
- アルゴリズム: PPO（連続行動）
- 使用ライブラリ : stable_baselines3
- 行動範囲: `x ∈ [-1, 1]`
- 学習長: `total_timesteps = 1,000,000`
- ログ: Weights & Biases + TensorBoard

---

# 実験1

- 画像入力よりもゲーム情報をそのまま使った方が早く終わりそう
- 上位50個のフルーツの`x,y,r,one-hot-id`
- `current_fruit_type`, `next_fruit_type`（one-hot）

<div class="flex items-center justify-center my-2">
  <img
    src="/assets/mlp.png"
    alt="mlp"
    class="rounded shadow"
    style="width: 60%;"
  />
</div>


基本設定
- 報酬: `merge score / 20` + `gameover penalty -2`
- PPO: `n_envs=16`, `n_steps=512`, `batch_size=1024`, `lr=1e-4`
- 行動: 連続値 `x ∈ [-1, 1]`

---

# 結果

- およそ100万ステップ（ローカルPCで６時間程度）
- エピソード報酬/最終スコアは改善傾向に

<div class="flex items-center justify-center gap-4 my-2">
  <img
    src="/assets/mlp_reward.png"
    alt="mlp reward"
    class="rounded shadow"
    style="width: 48%;"
  />
  <img
    src="/assets/mlp_score.png"
    alt="mlp score"
    class="rounded shadow"
    style="width: 48%;"
  />
</div>


ただ、端を積み上げるという局所解から抜け出せなかった

---

# MLP の挙動比較（初期 vs 最終）

- 点数は低いが、少しだけ方策の発露が見える

<div class="grid grid-cols-2 gap-4 items-center">
  <div class="text-center">
    <div class="mb-2 font-semibold">初期方策</div>
    <img
      src="/assets/mlp_first.gif"
      alt="mlp first policy"
      class="rounded shadow mx-auto"
      style="width: 90%; max-height: 360px; object-fit: contain;"
    />
  </div>
  <div class="text-center">
    <div class="mb-2 font-semibold">最終方策</div>
    <img
      src="/assets/mlp-fianl.gif"
      alt="mlp final policy"
      class="rounded shadow mx-auto"
      style="width: 90%; max-height: 360px; object-fit: contain;"
    />
  </div>
</div>

---

# 実験2

- 画像にしたらもう少し良くなるのでは？
- `64x64 RGBx1`(グレースケールだとフルーツの違いが分かりにくいので)
- `current_fruit_type`, `next_fruit_type`（one-hot連結）

<div class="flex items-center justify-center my-2">
  <img
    src="/assets/ppo-cnn.png"
    alt="mlp"
    class="rounded shadow"
    style="width: 60%;"
  />
</div>

基本設定
- 報酬: `merge score / 20` + `gameover penalty -2`
- PPO: `n_envs=16`, `n_steps=512`, `batch_size=1024`, `lr=3e-4`
- 行動: 連続値 `x ∈ [-1, 1]`

---

# 結果

- およそ100万ステップ（owl1で15時間程度,なぜかローカルの方が早い）
- エピソード報酬/最終スコアに改善傾向は見られなかった
- ステップ数の不足により画像の特徴量を捉えられなかった？

<div class="flex items-center justify-center gap-4 my-2">
  <img
    src="/assets/cnn-reward.png"
    alt="mlp reward"
    class="rounded shadow"
    style="width: 48%;"
  />
  <img
    src="/assets/cnn-score.png"
    alt="mlp score"
    class="rounded shadow"
    style="width: 48%;"
  />
</div>


ただ、中心に落とし続けることしかしなかった


---

# 実験3

- 画像よりは情報量を絞りつつも、単純なmlpよりも構造を捉えたい -> GNNの採用
- Node: `[x, y, radius_norm, one-hot(type)]`
- Edge: `[dx, dy, distance, is_sametype, is_touching, overlap_margin]`
- `k=8` + `distance < threshold` で疎グラフ化

<div class="flex items-center justify-center my-2">
  <img
    src="/assets/ppo-gnn.png"
    alt="mlp"
    class="rounded shadow"
    style="width: 50%;"
  />
</div>

基本設定
- モデル: Node encoder → 3層Residual GNN → mean/max pooling
- PPO: `n_envs=8`, `n_steps=512`, `batch_size=256`, `lr=1e-4`
- 行動: 連続値 `x ∈ [-1, 1]`

---

# 結果
- およそ100万ステップ（ローカルで12時間程度）
- エピソード報酬/最終スコアに改善傾向はほぼ見られなかった
- 学習が足りない可能性大


<div class="flex items-center justify-center gap-4 my-2">
  <img
    src="/assets/gnn-reward.png"
    alt="mlp reward"
    class="rounded shadow"
    style="width: 48%;"
  />
  <img
    src="/assets/gnn-score.png"
    alt="mlp score"
    class="rounded shadow"
    style="width: 48%;"
  />
</div>

---

# 結論

- 人間のスコアが2000~2500 >> RLの結果
- 学習が足りない可能性
- タスクが単純に難しい(取れる方策に対して状態数が非常に多い)

# 感想

- 報酬の設計が難しい
- サンプル数を稼ぐために環境のレイテンシが非常に重要