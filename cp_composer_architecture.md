# CP-Composer 아키텍처: Encoder → Latent Diffusion → Decoder → MCTG

코드 기준으로 **Encoder → Latent Diffusion → Decoder → MCTG** 흐름을 아키텍처 관점에서 정리한 문서입니다.

---

## 1. 전체 파이프라인 개요

```
[입력] receptor + ligand(peptide) 그래프 (X: [N,14,3], S: [N], mask: ligand=1)
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ENCODER (AutoEncoder, freeze)                                               │
│  X,S → H_0 (node hidden), Z (latent coords) → rsample → latent_H, latent_X  │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LATENT DIFFUSION (FullDPM / PromptDPM / MCTGPromptDPM)                      │
│  t=T 노이즈 상태 → EpsilonNet으로 ε 예측 → denoise step 반복 → t=0 잠재 상태  │
│  (MCTG 사용 시: 트리에서 expand → rollout → decode → score → selection/backprop) │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  DECODER (AutoEncoder.test, freeze)                                          │
│  latent_H, latent_X → seq_decoder → sequence; sidechain_decoder → coords    │
│  → backbone_model / sidechain_model (선택) → batch_X, batch_S                │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
[출력] 생성된 peptide 좌표 (N_gen, 14, 3), 시퀀스 문자열
      (+ MCTG 시: Pareto archive, history)
```

- **입력**: 한 배치의 단백질·펩타이드 **all-atom 3D 그래프** (노드 = residue, 채널 = 원자 좌표 N/CA/C/O + sidechain, 최대 14채널).
- **Encoder**: 이 그래프를 **잠재 시퀀스(latent_H)** 와 **잠재 좌표(latent_X, Z)** 로 압축.
- **Latent diffusion**: 잠재 공간에서 **H, X**에 대한 diffusion 역과정(denoising) 수행. MCTG 사용 시 트리 확장·롤아웃·스코어링·선택·백프로파게이션 포함.
- **Decoder**: denoising 끝난 **latent_H, latent_X**를 다시 **시퀀스 + all-atom 좌표**로 복원.

---

## 2. Encoder (AutoEncoder)

**파일**: `models/autoencoder/model.py` (클래스 `AutoEncoder`)

### 2.1 입력·그래프 구성

- **노드**: residue 단위. 각 노드에 `X` [n_channel, 3] (기본 14채널: N, CA, C, O + sidechain atoms), `S` (아미노산 인덱스).
- **엣지**: `prepare_inputs()`에서 `variadic_meshgrid`로 같은/다른 단백질 내 모든 노드 쌍 구성.
  - **ctx_edges**: 같은 단백질 내 (mask[row]==mask[col]).
  - **inter_edges**: 서로 다른 단백질(예: receptor–ligand) 간.
- **노드 피처**: `aa_feature(S, position_ids)` → `H_0` [N, embed_size], `atom_embeddings` [N, n_channel, atom_embed_size], `atom_weights`(atom_mask).

### 2.2 Encoder 본체

- **구조**: `create_encoder('dyMEAN', ...)` → **AMEGNN** (Adaptive Multi-channel EGNN).
  - 입력: `H_0`, `X`, `edges`, `channel_attr=atom_embeddings`, `channel_weights=atom_mask`.
  - 출력: `H` [N, hidden_size], `pred_X` [N, n_channel, 3].
- **마스킹**: 생성할 ligand만 사용. `H = H[mask]`.

### 2.3 잠재 공간으로의 투영

- **시퀀스 잠재 (H)**  
  - `W_mean`, `W_log_var`: Linear(hidden_size → latent_size).  
  - `rsample()`: `H_mean + exp(log_var/2)*noise` (학습 시), `noise` 없음 옵션(추론 시).  
  - KL 항: `H_kl_loss` (VAE 표준).

- **좌표 잠재 (Z)**  
  - `_get_latent_channels(X, atom_mask)`: 전체 원자 좌표를 **latent_n_channel**개 대표점으로 요약 (예: 1채널이면 무게중심, 5채널이면 bb 4 + 중심 1).  
  - `Z_centers = _get_latent_channel_anchors(true_X, atom_mask)` (prior 중심).  
  - `W_Z_log_var`: Linear(hidden_size → latent_n_channel*3).  
  - `rsample()`: `Z + exp(log_var/2)*noise` (학습 시).  
  - KL 항: `Z_kl_loss` (좌표 prior 대비).

- **최종 encoder 출력**: `latent_H` [N_gen, latent_size], `latent_X`(Z) [N_gen, latent_n_channel, 3]. (N_gen = mask.sum().)

---

## 3. Latent Diffusion (DPM)

**파일**: `models/LDM/diffusion/dpm_full.py`, `dpm_mctg.py`, `transition.py`

### 3.0 잠재 H와 X의 결합 방식 및 Diffusion 수식

#### 두 잠재 벡터를 “합치는” 방식

**합치지 않는다.**  
`latent_H`와 `latent_X`(Z)는 **한 개의 벡터로 concat하거나 하나의 스칼라/좌표로 합치지 않고**, **같은 네트워크(EpsilonNet → AMEGNN)에 두 개의 입력 스트림**으로 들어간다.

1. **스칼라 스트림 (시퀀스 잠재 H)**  
   - 노드 피처로 사용:  
     `in_feat = [H_noisy; t_embed; atom_gt; position_embedding]`  
     → `h` [N, in_node_nf]  
   - `t_embed = [β, sin(β), cos(β)]`, `β = get_timestamp(t)` (같은 t를 H·X 공유).

2. **좌표 스트림 (잠재 좌표 X)**  
   - 노드 좌표로 사용: `X_noisy` [N, 14, 3] (또는 latent 채널만 쓰는 경우에도 14채널로 채워서 입력).

3. **결합이 일어나는 곳**  
   - **AMEGNN** 한 개가 `(h, x)`를 동시에 받아, **같은 엣지·같은 레이어**에서 메시지 패싱:
     - 엣지 피처는 `X_noisy` 기준 거리로 계산 (RBF 등).
     - 메시지가 `h`를 업데이트하고, equivariant하게 `x`도 업데이트.
   - 따라서 “합침”은 **같은 GNN 안에서의 결합** (공유 메시지 패싱)이며, 출력은 다시 **두 갈래**:
     - `next_H` → `eps_H = next_H - H_noisy` (스칼라 노이즈)
     - `next_X` → `eps_X = next_X - X_noisy` (좌표 노이즈)

정리하면, **수식적으로 하나로 합치는 게 아니라**, **동일한 t에서 (H_t, X_t)를 같은 EpsilonNet에 넣고**, 네트워크가 **내부에서 H와 X를 함께 사용해** ε_H, ε_X를 각각 예측한다.

#### Diffusion에 적용되는 수식

H와 X **각각**에 **동일한 DDPM 공식**이 적용된다.  
같은 스케줄(같은 `t`, 같은 `ᾱ_t`)을 쓰고, **trans_h**와 **trans_x**는 별도 모듈이지만 수식은 동일하다.

- **Variance schedule (cosine)**  
  \( t = 0..T \), \( f_t = \cos^2\bigl(\frac{\pi}{2} \cdot \frac{t/T + s}{1+s}\bigr) \), \( \bar\alpha_t = f_t / f_0 \),  
  \( \alpha_t = 1 - \beta_t \), \( \beta_t = 1 - \bar\alpha_t/\bar\alpha_{t-1} \), \( \sigma_t \)는 improved DDPM 식으로 계산.

- **Forward (add_noise)** — 학습 시, **mask 영역만**  
  \[
  p_t = \sqrt{\bar\alpha_t}\, p_0 + \sqrt{1-\bar\alpha_t}\, \varepsilon,\quad \varepsilon \sim \mathcal{N}(0,I).
  \]  
  `p_0` 자리에는 `H_0` 또는 `X_0`가 들어가고, `p_t`는 `H_noisy` 또는 `X_noisy`.  
  context(mask 밖)은 `p_t = p_0` 유지.

- **Reverse (denoise)** — 샘플링 시, **mask 영역만**  
  \[
  p_{t-1} = \frac{1}{\sqrt{\alpha_t}}\bigl( p_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\, \varepsilon_\theta \bigr) + \sigma_t z,\quad z\sim\mathcal{N}(0,I)\text{ (t>1일 때만, t=1이면 }z=0\text{)}.
  \]  
  `ε_θ`는 EpsilonNet이 한 번에 예측한 `eps_H`, `eps_X` (각각 `trans_h.denoise`, `trans_x.denoise`에 넣음).  
  Optional: `guidance`가 있으면 `ε_θ := ε_θ - √(1−ᾱ_t)·guidance·weight` 후 위 식 적용.

- **적용 요약**  
  - **H**: `trans_h.add_noise(H_0, ...)` → `H_noisy`; `trans_h.denoise(H_t, eps_H, ...)` → `H_{t-1}`.  
  - **X**: `trans_x.add_noise(X_0, ...)` → `X_noisy`; `trans_x.denoise(X_t, eps_X, ...)` → `X_{t-1}`.  
  - 매 step에서 **같은 t**로 `eps_H`, `eps_X`를 한 번에 예측한 뒤, **각각** denoise하므로, 두 잠재는 **같은 스케줄·같은 step**으로 결합되지만, **수식은 각 도메인(H, X)에 독립 적용**된다.

---

### 3.1 LDM에서의 사용

- **Prompt_LDMPepDesign** (코드상 주 사용):  
  - Encoder 출력을 받아 **context 쪽은 고정**, **생성 영역(mask=1)만** diffusion.
  - `H_0`: context는 `aa_feature`로 채우고, mask 영역은 `hidden2latent(H_0)` 후 mask 위치에 **latent_H** 삽입 (학습 시), 샘플링 시에는 0 또는 랜덤.
  - `X_0`: context는 원래 좌표, mask 영역은 `_fill_latent_channels(Z)` 또는 0/랜덤.
  - `position_embedding`: `abs_position_encoding(position_ids)` [N, latent_size].
- **Diffusion 모듈**: `FullDPM` 또는 `PromptDPM` (및 MCTG 시 `MCTGPromptDPM`).

### 3.2 Forward (학습)

- **시간**: `t ~ Uniform(0, T)`, T = num_steps (예: 100).
- **Transition**: `ContinuousTransition` (cosine schedule).
  - `add_noise`: `p_noisy = sqrt(alpha_bar)*p_0 + sqrt(1-alpha_bar)*eps`, **mask 영역만** 노이즈 적용, context는 p_0 유지.
- **정규화**: `_normalize_position(X, batch_ids, mask_generate, atom_mask, L)`  
  - context CA로 batch별 중심(centers) 계산, L(Cholesky) 있으면 공분산 정규화.
- **EpsilonNet**  
  - 입력: `H_noisy`, `X_noisy`, `position_embedding`, `ctx_edges`, `inter_edges`, `atom_embeddings`, `atom_mask`, `mask_generate`, `beta=t_embed`, (PromptDPM 시) `atom_gt`, `guidance_edges`, `guidance_edge_attr`.  
  - `in_feat = [H_noisy; t_embed; atom_gt; position_embedding]` → **AMEGNN** (또는 Prompt_AMEGNN) → `next_H`, `next_X`.  
  - `eps_H = next_H - H_noisy`, `eps_X = next_X - X_noisy`, **mask 밖은 0으로**.
- **Loss**: `mask` 영역만 MSE(eps_pred, eps_true). 시퀀스/구조 각각 스케일 가중.

### 3.3 Sample (추론, t=T→0)

- **초기화**: `X_init`, `H_init`에서 **mask 영역만** 랜덤 노이즈, context는 원본 유지.
- **반복** (t = T → 1):  
  - `ctx_edges`, `inter_edges` (및 Prompt 시 `sampled_edges`, `guidance_edge_attr`) 계산.  
  - EpsilonNet으로 `eps_H`, `eps_X` 예측 (Prompt 시 classifier-free: uncond + cond, `eps = (1+w)*eps_cond - w*eps_uncond`).  
  - `trans_h.denoise`, `trans_x.denoise`: DDPM 업데이트, **mask 영역만** 갱신, context는 그대로.  
  - (선택) energy_func 있으면 gradient guidance로 eps 수정.
- **Transition.denoise**:  
  - `p_next = c0*(p_t - c1*eps) + sigma*z`, z는 t>1일 때만 노이즈.  
  - `p_next = where(mask, p_next, p_t)`.
- **최종**: t=0의 `H`, `X` (unnormalize된 좌표)를 Decoder로 넘김.

### 3.4 EpsilonNet 상세

- **입력 차원**: `enc_input_size = latent_size + 3 (t_embed) + 20 (atom_gt) + latent_size (position_embedding)` 등.
- **Encoder**: AMEGNN (또는 Prompt_AMEGNN) — 메시지 패싱 + equivariant 좌표 업데이트.
- **출력**: `hidden2input` Linear로 `next_H` → 스칼라 차원 맞춤 후 `eps_H = next_H - H_noisy`, `eps_X = next_X - X_noisy`.

---

## 4. Decoder (AutoEncoder.test)

**파일**: `models/autoencoder/model.py` — `decode()`, `_reconstruct()`, `test()`

### 4.1 Decoder 입력

- **given_laten_H**, **given_latent_X**: Diffusion이 t=0에서 준 잠재 벡터/좌표 (또는 encoder 직접 출력).
- **X, S, mask**: 전체 그래프; mask 영역을 decoder가 채움.

### 4.2 decode() 내부

- **latent → 그래프 채우기**  
  - 구조: `X[mask] = _fill_latent_channels(Z)` (latent 1채널이면 모든 원자 채널에 동일 복사).  
  - 시퀀스: `S[mask] = latent_id` (플레이스홀더), `H_from_latent = latent2hidden(H)`.
- **엣지**: 동일한 `prepare_inputs` → ctx_edges, inter_edges.
- **시퀀스 디코더**  
  - `H_0 = aa_feature(S, position_ids)`, mask 위치만 `H_from_latent`로 교체.  
  - **seq_decoder** (AMEGNN): `H_0`, `X`, `edges`, channel_attr/weights → `H`.  
  - `pred_S_logits = seq_linear(H[mask])` → `S[mask] = s_remap[argmax(logits)]`.
- **사이드체인 디코더**  
  - `H_0 = aa_feature(S, position_ids)` (갱신된 S 기준), mask 쪽은 `merge_S_H([H_from_latent; H_0])`.  
  - **sidechain_decoder** (AMEGNN): `H_0`, `X`, `edges` → `pred_X`.  
  - `pred_X = pred_X[mask]` → residue별 all-atom 좌표.
- **(선택) idealize 시**: `backbone_model`, `sidechain_model`로 backbone/chi 정규화.

### 4.3 test() 반환

- `batch_X`: 샘플별 생성 residue의 좌표 리스트 [ (n_res, 14, 3) ].
- `batch_S`: 샘플별 1-letter 시퀀스 문자열.
- `batch_ppls`: 샘플별 perplexity.

---

## 5. MCTG (Monte Carlo Tree Search for Molecule Generation)

**파일**: `models/LDM/diffusion/dpm_mctg.py`, `mctg_runner.py`

MCTG는 **latent space에서 트리 탐색**으로 여러 후보를 만들고, rollout → decode → scoring → Pareto 보관·선택·백프로파게이션을 반복합니다.

### 5.1 컨텍스트 준비

- **prepare_mctg_context**:  
  - LDM 입력(H, X, position_embedding, mask, lengths, atom_embeddings, atom_mask, L 등)으로 **정규화·초기 노이즈** 설정.  
  - `X_init`, `H_init`: mask만 랜덤, 나머지 context 고정.  
  - 반환 dict를 `dctx`로 사용.
- **init_root_state**: `{ "t": T, "X": X_init, "H": H_init }` = 루트 노드 상태.

### 5.2 Expansion (expand_children)

- **부모 상태**: `parent_state["t"]`, `parent_state["X"]`, `parent_state["H"]`.
- **자식 수**: `num_children` (예: 8).
- **각 자식**:
  - `child_state = copy(parent_state)`.
  - **mask 영역만** `child_noise_scale * randn` 추가 (표준 가우시안).  
    → PepTune과 달리 **같은 t에서 여러 방향으로 분기**.
  - `sample_prob = exp(-0.5 * (noise_x^2 + noise_h^2))` (surrogate).
  - **한 스텝 denoise**: `_denoise_step(child_state, ctx)` → t가 1 줄어든 상태.
- **반환**: `children` 리스트 (각각 `t`, `X`, `H`, `sample_prob`).

### 5.3 Rollout (rollout_children)

- **입력**: 위에서 만든 `child_states` (각각 t=T-1 등).
- **동작**: 각 child에 대해 **t가 0이 될 때까지** `_denoise_step` 반복 (같은 ctx, 동일 EpsilonNet/transition).
- **출력**: t=0, unnormalize된 `X`, `H` 리스트 = **rolled**.

### 5.4 Decode & Score

- **Decode**: `_decode_from_latent(decode_ctx, rolled[i])`  
  → `autoencoder.test(given_laten_H=rolled[i]["H"], given_latent_X=rolled[i]["X"][mask][:,:latent_n_channel])`  
  → `batch_X`, `batch_S`, `batch_ppl`.
- **시퀀스/좌표**: `seqs`, `decoded_coords` 리스트.
- **SDF (선택)**: `seq_to_sdf_fn(seq, meta)` — `decoded_coords` + condition_id로 cyclization 적용 후 SDF 저장.
- **Score**: `scorer(sdf_paths)` 또는 `scorer(smiles)` → `score_vectors` [num_children, num_objectives], `valid_mask`.

### 5.5 Selection (트리 상에서)

- **노드 상태**:  
  - 1 = terminal (t≤0), 2 = expandable leaf (자식 없음), 3 = non-leaf.
- **Selection**: root에서 시작해 **U-score**로 하향 이동.  
  - `U = W/N_visit + c * sample_prob * sqrt(N_parent) / (1 + N_child)`.  
  - 같은 부모의 자식들은 **Pareto front(active objectives)** 로 non-dominated 필터 후, first front에서 uniform random 선택.
- **종료**: terminal leaf가 선택되면 (그리고 root면) 루프 종료.

### 5.6 Backpropagation & Pareto

- **Reward**: PepTune 스타일.  
  - `reward[k] = (현재 Pareto archive 중 score_k >= 해당 Pareto 항목인 개수) / |Pareto|`.  
  - 새 후보가 active objectives 기준 non-dominated면 Pareto set에 추가, 지배당한 기존 항목 제거.
- **Backprop**: 선택된 leaf부터 root까지 올라가며 `total_reward += reward`, `visits += 1`.
- **invalid_penalty**: valid하지 않은 자식 비율만큼 reward에서 차감.

### 5.7 Final 반환

- **마지막 iteration**에서 **best child** (Pareto first front에서 선택된 child) 상태를 한 번 더 **rollout** (t→0) 후 **decode**.
- **반환**: `final` (X, S, ppl), `history` (iter별 valid_children, best_seq, scores, pareto_fronts 등), `pareto_archive`.

---

## 6. 데이터/차원 요약

| 단계           | 입력/출력 | 차원·비고 |
|----------------|-----------|-----------|
| Encoder 입력   | X         | [N, 14, 3] (또는 4 if backbone_only) |
| Encoder 입력   | S         | [N] (residue type index) |
| Encoder 출력   | latent_H  | [N_gen, latent_size] |
| Encoder 출력   | Z         | [N_gen, latent_n_channel, 3] (예: 1 or 5) |
| Diffusion H    | H_0, H_t  | [N, latent_size] (mask 영역만 갱신) |
| Diffusion X    | X_0, X_t  | [N, 14, 3], mask 영역은 latent_n_channel만 사용 후 _fill로 14채널 |
| EpsilonNet     | eps_H     | [N, latent_size], eps_X [N, 14, 3] (mask 밖 0) |
| Decoder 출력   | recon_X   | [N_gen, 14, 3], recon_S [N_gen] → 1-letter seq |

- **엣지**: ctx_edges + inter_edges (풀 메시 그래프), PromptDPM 시 augmented k-hop/head-tail edges로 guidance_edge_attr 추가.
- **Transition**: cosine variance schedule, ContinuousTransition (DDPM 스타일 add_noise/denoise).

이 문서는 `CP-Composer_final` 코드 구조를 반영한 Encoder → Latent Diffusion → Decoder → MCTG의 세부 아키텍처 정리입니다.
