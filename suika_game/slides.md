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

# 実験1


入力
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

# 実験2

入力
- `64x64 RGB`（1フレーム）
- `current_fruit_type`, `next_fruit_type`（one-hot連結）

<div class="flex items-center justify-center my-2">
  <img
    src="/assets/ppo-cnn.png"
    alt="mlp"
    class="rounded shadow"
    style="width: 60%;"
  />
</div>

基本設定（今回の実装）
- 報酬: `merge score / 20` + `gameover penalty -2`
- PPO: `n_envs=16`, `n_steps=512`, `batch_size=1024`, `lr=3e-4`
- 行動: 連続値 `x ∈ [-1, 1]`

---

# 実験3

入力グラフ
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

基本設定（今回の実装）
- モデル: Node encoder → 3層Residual GNN → mean/max pooling
- PPO: `n_envs=8`, `n_steps=512`, `batch_size=256`, `lr=1e-4`
- 行動: 連続値 `x ∈ [-1, 1]`
