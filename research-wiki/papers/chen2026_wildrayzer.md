---
type: paper
node_id: paper:chen2026_wildrayzer
title: "WildRayZer: Self-supervised Large View Synthesis in Dynamic Environments"
authors: ["Xuweiyi Chen", "Wentao Zhou", "Zezhou Cheng"]
year: 2026
venue: "CVPR 2026 (Highlight)"
external_ids:
  arxiv: "2601.10716"
tags: [dynamic-scene, visibility, gradient-masking, transient, self-supervised]
status: deep-dived
---

# WildRayZer: Self-supervised Large View Synthesis in Dynamic Environments

**Paper:** https://arxiv.org/abs/2601.10716
**Code:** https://github.com/UVA-Computer-Vision-Lab/wild-rayzer
**Base method:** RayZer (Jiang et al., ICCV 2025, Best Student Paper Honorable
Mention) — a pose-free, self-supervised feedforward transformer for static
large view synthesis (camera estimator + scene encoder + rendering decoder).
Also builds on DINOv3 (frozen ViT backbone) for pseudo-label features, and
classical CV primitives (K-means, SSIM, GrabCut) for pseudo-mask refinement.

## One-line thesis

A frozen static-only RayZer renderer's own reconstruction residuals, fused
with DINOv3 semantic dissimilarity and SSIM appearance dissimilarity under a
training-stage-adaptive weighting, are clustered into pseudo motion masks that
distill a learned motion estimator; that estimator then zeroes dynamic input
tokens before the scene encoder and gates reconstruction losses to static
pixels, letting a pose-free feedforward transformer learn multiview-consistent
static geometry from real dynamic video with zero ground-truth pose, depth, or
mask supervision.

## Problem / Gap

RayZer and LVSM-style feedforward view-synthesis transformers "critically rely
on a fundamental assumption: the 3D scene is static" — any moving content
violates the multiview photometric consistency the scene encoder assumes,
producing ghosting and hallucinated geometry when the model tries to fold
transient pixels into a single rigid scene representation. Separately, no
large-scale real-world dynamic multiview dataset exists to train around this:
prior dynamic datasets (DyCheck, WildGS-SLAM, etc.) top out at roughly 10-17
sequences, too small to train a feedforward model at RayZer's scale.

## Method

Augments RayZer with a fourth module, a motion estimator ℰ_mot, trained in
three alternating stages on top of a frozen RayZer backbone pretrained on
static RealEstate10K. Stage 2 freezes the renderer stack and trains ℰ_mot
against pseudo-labels built purely from analysis-by-synthesis: the static
renderer's own residuals (DINOv3 patch dissimilarity + SSIM dissimilarity,
z-scored and fused) are clustered across frames of a scene and refined into
binary masks. Stage 3 freezes the motion head and trains the renderer on a
scene encoder whose input tokens have had dynamic patches zeroed out
(MAE-style token dropping), so transient content never enters the persistent
scene representation `z`. A final joint fine-tuning stage adds COCO copy-paste
augmentation, pasting segmented objects onto training images to supply exact
ground-truth transient masks that override the model's own noisy predictions,
broadening the motion estimator's coverage of unseen object types.

## Assumptions

Multiple (2-4 evaluated) unposed, uncalibrated RGB input views of the same
indoor scene with sufficient camera translation/parallax (curated via
translation-magnitude subdivision); no ground-truth pose, depth, or dynamic
mask is required at train or test time. The analysis-by-synthesis pseudo-label
route implicitly assumes the scene is majority-static so that a static-only
renderer can first produce a mostly-correct reconstruction whose residuals are
informative rather than dominated by noise.

## Limitations / Failure Modes

The paper states that in its sparse-view regime (2-4 input views), "reliable
evaluation requires substantial spatial overlap between inputs and targets;
otherwise, reconstruction errors become entangled with coverage gaps rather
than modeling quality" — i.e., the method's own evaluation protocol can
confound missing-view coverage with genuine dynamic-region failure. The
released model is view-count-locked: the README states "the model can be
sensitive to the number of views due to image index positional embedding" and
requires matching train/test view counts. On D-RE10K-iPhone at 2 views (Table
3), transient-region LPIPS (0.371) still trails static-region LPIPS (0.360)
despite being the best method in the comparison, indicating incomplete parity
between static and transient reconstruction quality even for WildRayZer
itself. A dedicated failure-case appendix (C.2) is referenced but its content
was not retrievable from the fetched HTML.

## Reusable Ingredients

- **Analysis-by-synthesis pseudo-labeling**: use the residual between a
  frozen static-only renderer's own self-reconstruction and the real captured
  image as the primary dynamic-region evidence source, instead of relying
  solely on an external motion/flow/segmentation model.
- **Z-scored multi-cue saliency fusion**: combine DINOv3 semantic
  dissimilarity and SSIM appearance dissimilarity via per-cue z-score
  normalization plus a training-stage-adaptive weight, so an unreliable early
  cue (photometric residual, before the renderer is any good) is automatically
  downweighted relative to a more stable cue (semantic dissimilarity).
- **Cross-frame patch-level K-means clustering**: cluster patch embeddings
  across multiple frames of the same scene and require both a top-percentile
  mean saliency and consistent salience in several frames before marking a
  cluster as foreground — a cheap (patch-resolution) way to get temporally
  consistent binary masks instead of per-frame independent thresholding.
- **Pre-scene-encoder token dropping**: mask out dynamic tokens (MAE-style)
  before they reach the persistent scene representation, so transient content
  is architecturally excluded rather than merely down-weighted in a loss.
- **Cross-domain copy-paste supervision**: paste segmented objects from an
  unrelated dataset (COCO) into training scenes and use the pasted region's
  exact mask as ground truth to override noisy self-distilled pseudo-labels,
  improving generalization to unseen dynamic-object categories.
- **PSNR-gated masked-loss fallback**: skip or soften pseudo-label-derived
  losses for the noisiest reconstructions, and fall back to the unmasked loss
  when too few static pixels remain after masking, to prevent training
  collapse from unreliable masks early in training.

---

### Deep Dive

#### Core Novelty

Relative to RayZer, the change is architectural placement plus supervision
source: a motion estimator sits between tokenization and the scene encoder,
consumes the renderer's own pre-camera-estimator tokens (never target-view or
test-time-only signals) plus DINOv3 features and Plücker rays, and is trained
against labels mined from the renderer's own analysis-by-synthesis residuals
rather than from an external motion model. The key insight is that "what a
static multiview renderer cannot explain" is a self-consistent, free
supervisory signal for exactly the content that would corrupt that renderer's
scene representation — closing the loop between the failure mode and its own
detector without any labeled data.

#### Mathematical Formulation

Base RayZer render loss (unchanged, evaluated per held-out target view after
rendering):
$$\mathcal{L} = \frac{1}{|\mathcal{I}_\mathcal{B}|}\sum \big[\mathrm{MSE}(I,\hat{I}) + \lambda \cdot \mathrm{Percep}(I,\hat{I})\big], \quad \lambda = 0.2$$

Pseudo-label saliency, computed offline/periodically from the frozen static
renderer's self-reconstruction residual against the real image, at patch
resolution:
$$D_{\mathrm{DINO}}(p) = 1 - \langle \Phi_p(I), \Phi_p(\hat{I}) \rangle$$
where $\Phi_p(\cdot)$ are L2-normalized DINOv3 patch features at patch $p$.

$$D_{\mathrm{SSIM}}(x) = 1 - \mathrm{SSIM}(I,\hat{I})(x)$$
computed per-pixel then area-pooled down to the patch grid.

Both cues are z-score normalized, $\mathcal{Z}(D) = (D-\mu_D)/\sigma_D$, and
fused with a training-stage-adaptive weight:
$$D_{\mathrm{bin}}(p) = w_{\mathrm{DINO}}\cdot\mathcal{Z}(D_{\mathrm{DINO}}(p)) + w_{\mathrm{SSIM}}\cdot\mathcal{Z}(D_{\mathrm{SSIM}}(p)), \quad w_{\mathrm{DINO}}+w_{\mathrm{SSIM}}=1$$
with $w_{\mathrm{DINO}}$ larger early in training (renderer output still
coarse, semantic cue more trustworthy) and $w_{\mathrm{SSIM}}$ increasing as
photometry improves. This fused saliency is evaluated once per training
sample, before the K-means clustering step that produces the binary pseudo-
label $\tilde{M}(I)$.

Cluster $k$ (from K-means over patch embeddings pooled across $B$ frames of
one scene) is labeled foreground/dynamic iff:
- mean saliency $\bar{s}_k = \mathbb{E}_{p\in k}[D_{\mathrm{bin}}(p)]$ is in
  the top 5% across clusters, AND
- the cluster is salient (above the 75th percentile of $D_{\mathrm{bin}}$) in
  at least 4 frames.

Motion estimator training loss (Stage 2, renderer frozen): standard BCE-with-
logits between the predicted per-pixel logit map $S(I)$ and the pseudo-label
$\tilde{M}(I) \in [0,1]^{H\times W}$.

Combined loss used in Stage 3 joint fine-tuning (per released code, not fully
spelled out as a single equation in the paper text): a weighted sum of masked
L2, LPIPS, perceptual, and motion-mask BCE distillation terms, with the
reconstruction terms computed only over patches whose predicted motion
probability is below a threshold.

#### Algorithm / Pipeline Changes

1. Tokenize each of $K$ unposed input images into non-overlapping $16\times16$
   patches at $256\times256$ resolution ($h=w=16$ token grid); these
   pre-camera-estimator tokens feed both the normal RayZer path and the motion
   estimator.
2. Camera Estimator $\mathcal{E}_{\mathrm{cam}}$ runs unchanged from RayZer,
   predicting per-view SE(3) poses and shared intrinsics.
3. Motion Mask Predictor $\mathcal{E}_{\mathrm{mot}}$ runs before the Plücker
   embedding is reshaped for the scene encoder: it projects (a) DINOv3 patch
   features (interpolated to the $16\times16$ grid), (b) the pre-camera-
   estimator image tokens, and (c) Plücker-ray tokens derived from the
   predicted $(\mathbf{P},\mathbf{K})$, each via LayerNorm + linear to a
   shared width $d_{\mathrm{fused}}$, concatenates them ($3\times d_{\mathrm{fused}}$),
   and fuses via an MLP back to $d_{\mathrm{fused}}$. A 4-block transformer
   (with QK-normalization and gradient checkpointing) reasons jointly over
   tokens concatenated across all input views for cross-view consistency. A
   DPT-style decoder with 5 sequential `Up2x` modules upsamples
   $16\to32\to64\to128\to256$ to a per-pixel logit map, output shape
   `[B*V, 1, 256, 256]`.
4. Dynamic token masking: the motion probability map is downsampled back to
   the $16\times16$ token grid; patches whose score exceeds a threshold are
   zeroed out in the fused input tokens before they reach the Scene Encoder —
   an MAE-style token-dropping step that replaces RayZer's original
   "feed all patch tokens to the scene encoder" step.
5. Scene Encoder $\mathcal{E}_{\mathrm{encode}}$ (architecture otherwise
   unchanged from RayZer) now consumes only the masked/static token set plus
   camera conditioning to produce $L=768$ scene tokens $z\in\mathbb{R}^{L\times d}$.
6. Rendering Decoder $\mathcal{D}_{\mathrm{render}}$ synthesizes held-out
   target views from $z$ and tokenized Plücker rays of the target camera —
   unchanged from RayZer.
7. Pseudo-label mining (feeds Stage 2 training): static-renderer residuals
   (DINO + SSIM, z-scored and fused) are clustered per-scene with K-means
   across frames, then refined by nearest-neighbor upsampling to pixel
   resolution, morphological smoothing, small-connected-component removal,
   and GrabCut boundary refinement seeded by the eroded foreground mask.
8. Copy-paste augmentation (Stage 3 only): COCO instance-segmented objects are
   pasted onto training images at random scale/position; the pasted region's
   ground-truth mask overrides the model's predicted motion score there
   during loss computation.

Three training stages overall: (1) RayZer backbone pretraining on static
RealEstate10K; (2) motion-mask training with the renderer frozen; (3) joint
fine-tuning of all components with copy-paste augmentation added.

#### Key Hyperparameters & Design Choices

From the paper text:
- Perceptual loss weight (base RayZer term): $\lambda = 0.2$.
- Learning rates: RayZer pretraining $4\times10^{-4}$; motion-mask training
  $2\times10^{-4}$; masked renderer training $1\times10^{-4}$; cosine schedule
  over 100k iterations.
- Batch size 64; 4× H100 for reported training runs.
- Architecture (paper): 28 transformer layers total, split 4 (motion
  estimator) / 8 (camera estimator) / 8 (scene encoder) / 8 (renderer); 768
  scene tokens; $256^2$ resolution; patch size 16.
- Cluster foreground criterion: top 5% mean saliency, >75th-percentile
  saliency in ≥4 frames (see Mathematical Formulation).

From the released configs (not stated numerically in the paper text — see
Implementation Reality for discrepancies against the paper's own numbers):
- Stage 2 (`wildrayzer_stage2_motion_mask.yaml`): LR $1\times10^{-4}$, 10,000
  steps, batch size 16/GPU, warmup 1,000 steps, gradient clip norm 1.0,
  AdamW $\beta=(0.9,0.95)$, weight decay 0.01; mask-distill loss weight 1.0,
  L2/LPIPS/perceptual weights all 0 (motion head only, renderer frozen);
  $w_{\mathrm{DINO}}=0.6$, $w_{\mathrm{SSIM}}=0.4$ (single fixed pair exposed
  in this config — the paper's adaptive schedule was not visible in the
  fetched file); PSNR filter threshold 17.0; K-means $K=64$.
- Stage 3 (`wildrayzer_stage3_joint_copy_paste.yaml`): LR $2\times10^{-4}$,
  20,000 steps, batch size 8/GPU, same warmup/clip/AdamW settings; loss
  weights L2 1.0, LPIPS 0.0, perceptual 0.2 (matches paper's $\lambda=0.2$),
  mask-distill 1.0; motion-mask binary threshold 0.1; copy-paste enabled with
  paste probability 0.8, 1-2 objects/scene, 18 COCO categories, scale range
  0.25-0.35, position margin 0.15.
- Motion Mask Predictor: 4 transformer blocks, $d_{\mathrm{dino}}\approx1024$
  (ViT-L DINOv3), DPT decoder with 5 `Up2x` stages.

If a value is not stated anywhere in the paper or the fetched configs (e.g.
the exact z-score-weight adaptation schedule/curve as a function of training
step), it is "Not specified."

#### Ablation Summary

Table 5 (copy-paste augmentation effects; values read as motion-mask mIoU,
matching the scale of the main mIoU results — the extracted table did not
carry an explicit unit label, so this reading should be treated as probable
rather than certain):

- Copy-Paste Only: 18.2 (D-RE10K) / 11.1 (D-RE10K-iPhone)
- Pseudo-Mask Only: 53.9 / 45.3
- Copy-Paste + Pseudo-Mask: 53.9 / 49.7

Finding: copy-paste alone does not transfer to real dynamic video (18.2/11.1),
confirming pseudo-mask pretraining from real analysis-by-synthesis residuals
is the necessary component. Adding copy-paste on top of pseudo-mask
pretraining is a pure generalization gain — no change on in-distribution
D-RE10K (53.9 → 53.9) but +4.4 mIoU on the harder/out-of-distribution
D-RE10K-iPhone set (45.3 → 49.7), and a much larger reported jump on DAVIS
cross-dataset transfer (mIoU 3.4 → 31.0).

Motion estimator input-modality ablation (reported in prose, not a formal
table): removing DINOv3 features from the fusion delays mask emergence from
~1.5k to ~20k steps to reach mIoU=30, and lowers final quality from 39.4 to
29.4 mIoU. **This is the single most impactful ablated component** — it
affects both final quality (+10.0 mIoU) and training efficiency (~13x fewer
steps to a fixed quality bar), a larger effect than copy-paste augmentation's
in-domain-neutral, generalization-only contribution.

#### Implementation Reality

- **Framework:** PyTorch, distributed training via `torchrun`; the RayZer
  backbone appears reimplemented in-repo (three versioned files
  `rayzer_official.py`, `_v2.py`, `_v3.py`) rather than imported as an
  external dependency, with `v3`'s `Images2Latent4D` class wired into the
  released Stage 3 config. Uses `xformers` for attention, requiring GPU
  compute capability > 8.0 (Ampere or newer).
- **Key files:** `model/rayzer_official_v3.py` (main model class
  `Images2Latent4D`, the `MotionMaskPredictor` class, and a
  `DinoV3UncertaintyPseudoLabelMaker` class implementing the DINO+SSIM fusion
  and K-means co-segmentation); `model/loss.py` (`PerceptualLoss`,
  `LossComputer`, `LossComputer_official`); `model/transformer.py` (shared
  transformer blocks); `configs/wildrayzer_stage{1,2,3}_*.yaml`; `train.py`,
  `inference.py`, `generate_html.py`.
- **Notable implementation details not stated in the paper, or apparently in
  tension with it:**
  - The motion-mask binary threshold used both for token masking and for
    masking reconstruction losses is 0.1 in code; the paper only says
    "patches whose score exceeds a threshold," with no numeric value.
  - A PSNR filter threshold of 17.0 (samples below this are excluded from
    pseudo-label-supervised losses, except for the static RE10K set) appears
    only in the config, not in the paper text.
  - `LossComputer_official` falls back to the unmasked loss if fewer than 10%
    of pixels remain static after masking — an unpublished training-stability
    guard.
  - `PerceptualLoss` is VGG19-based with max-pooling replaced by average
    pooling, six feature blocks at depths `[0, 4, 9, 14, 23, 32]`, and
    per-block scaling factors ($e_1/2.6$, $e_2/4.8$, $e_3/3.7$, $e_4/5.6$,
    $e_5\times10/1.5$, final sum divided by 255) — none of this is described
    in the paper's math.
  - Copy-paste augmentation's numeric parameters (0.8 paste probability,
    0.25-0.35 scale, 0.15 margin, 18 COCO categories, 1-2 objects/scene) are
    fully specified in config but not quantified in the paper.
  - Apparent paper-vs-config mismatches that should be treated as unresolved
    rather than confirmed errors (extracted via automated page fetches, not a
    manual byte-level diff of the LaTeX source): the paper states 8 layers
    each for the camera estimator and renderer (28 total with 4+8+8+8), while
    the fetched Stage 2/3 configs report 12 layers for the pose-estimation
    encoder and 12 for the render decoder; the paper states motion-mask-stage
    LR $2\times10^{-4}$, while the fetched Stage 2 config shows
    $1\times10^{-4}$.
  - Only a single 2-input-view checkpoint is released (via Hugging Face, not
    bundled in the repo); the README explicitly warns the positional
    embedding scheme is view-count-locked between training and inference.

#### Failure Modes & Limitations

The paper's own stated caveat is about evaluation confounding rather than a
model failure per se: in the 2-4 view sparse regime, insufficient input-target
overlap makes reconstruction error reflect missing coverage rather than
modeling quality. A dedicated "C.2 Failure Cases" appendix section is
referenced in the paper's table of contents but its content was not
accessible from the fetched HTML, so specific qualitative failure categories
(e.g. reflective/transparent surfaces, extreme motion blur) could not be
confirmed from the paper itself — do not attribute those to the paper without
direct verification. The quantitative results do show a residual static/
transient gap even for WildRayZer itself: on D-RE10K-iPhone at 2 views,
transient-region LPIPS (0.371) and SSIM (0.575) both trail static-region
values (0.360 / 0.612) despite WildRayZer having the smallest such gap among
compared methods.

---

## Relevance to ADAGS

Strongest recent occupant of visibility-gated gradient masking (protection
by gradient gating) — must be distinguished from any ADAGS protection or
exposure-weighting claim: WildRayZer gates supervision by learned transient
masks in a feedforward large-view-synthesis setting; ADAGS protection gates
per-primitive updates from per-view occlusion state in per-scene multiview
optimization.

## Connections

- Pressures [[gap_map#G13 - Visibility Events Are Not Smooth Deformation]]
- Pressures [[gap_map#G9 - Uncertainty And Occlusion Confidence Are Underused In ADAGS]]

## Sources

- https://arxiv.org/abs/2601.10716
- https://github.com/UVA-Computer-Vision-Lab/wild-rayzer
