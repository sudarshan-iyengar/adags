# FROZEN — LRV4, the observation-starved fixture: prediction and
# decision rule (2026-08-23)

Status: **FROZEN before any LRV4 result exists.** EXPLORATORY,
`evidence_bearing: false`. Substrate cell: Determined experiment **247**
(`lrv4_b1_packets` r0, commit `bbf1c4f`, dgx, admitted V100 image, seed
default 6666 to match LRV3's substrate experiment 209).

## 1. The claim under test

[[payload-headroom-result-2026-08-23]] measured that no transferable
per-row quantity has headroom on LRV3, and explained it with a mechanism
claim that the page labels INFERENCE, not measurement:

> LRV3's returning surface is identical in pose, colour and texture and
> is observed by 48 training view-frames, so the recipient rows are wrong
> about NOTHING. **Headroom is a question about OBSERVATION SUPPLY, not
> about which tensor is carried.**

**LRV4 is the falsification test of that specific claim.** It is not a
retry of the payload search, and this page exists so that framing is on
the record before any number is read.

## 2. The fixture — exactly one variable

LRV4 is LRV3 with a **one-frame return** (frame 59 only) instead of
three (57-59). Everything else is identical: same 20 cameras and same 4
held out, same 400×300 raster, 60 frames at 6 fps, same object centre,
radius, colour and texture, same lighting, same ground extent, same
seed, same initialization cloud.

Verified structurally: `transforms_train.json`, `transforms_test.json`
and `points3d.ply` are **byte-identical** to LRV3's, and on cam02 exactly
**58 of 60 frames are identical** — the two that differ are frames 57 and
58, precisely where the object is present in LRV3 and absent in LRV4.

**The measured variable:** held-out return supply drops from
**56,934 pixel-times to 18,978 — exactly one third**. Training-view
supply for the return drops from 48 view-frames to 16.

LRV3 regeneration remains byte-identical under the generator's default
flags, proven against the fixture on disk (manifest sha256
`c36cdb14…f7b16a` over 1,444 files from three independent sources).

**LRV4 is INADMISSIBLE for gated-presence experiments** — a one-frame
return cannot clear `floor_len`, and the generator refuses loudly in
three places. It is a PAYLOAD fixture only.

## 3. The frozen prediction

If the mechanism claim is true, starving the return by 3× should raise
the recipient rows' error — they now have a third of the evidence — and
therefore **open headroom** between the recipient's trained values and
the donor's.

**Prediction, recorded before the run: at least one candidate tensor's
`headroom_ratio` rises materially above its LRV3 value**, with
`_features_dc` (LRV3: 1.429) and `_opacity` (LRV3: 1.223 activated,
1.257 raw) the most likely, since both are photometric and directly
starved by fewer observations. Geometry should move least: `_xyz`
(LRV3: 1.092) is anchored by the shared initialization cloud and by
episode 1, which is byte-identical between the fixtures.

## 4. The decision rule, frozen

* **CLAIM SURVIVES** if any candidate tensor's `headroom_ratio` on LRV4
  exceeds its LRV3 value by a margin larger than the same-identity floor
  can explain, AND that tensor's `discrimination_ratio` does not
  collapse. The payload edit then becomes worth running on LRV4 with the
  full control set — **including L4**, which is now mandatory rather than
  optional (see §5).
* **CLAIM REFUTED** if every tensor's headroom ratio stays within the
  LRV3 band (0.77-1.43). That would mean a 3× reduction in observation
  supply does not open headroom, so the LRV3 negative is NOT explained by
  observation sufficiency, and **the representation-only pivot hardens
  from a recommendation into a finding**.
* **Either outcome is a result.** Neither licenses any N3V claim.

## 5. L4 is mandatory on any LRV4 edit, and this is why

Experiment 244 established that on LRV3 the opacity edit's held-out
damage is **monotone in edit magnitude and independent of identity
correctness**: destroying identity cost −0.97 dB at 0.6% of the oracle
link's displacement and half its edit volume. **Any future edit on any
fixture must therefore run the within-recipient permutation control
alongside the oracle link**, or its result cannot be attributed to
identity. Recorded as a standing requirement, not a per-experiment
option.

## 6. Integrity check on the output — a real failure mode

The falsification tool's fixture constants are now derived from
`event_spec.json` rather than hardcoded, and LRV3's frozen probe time of
**9.6 s falls inside LRV4's absence gap**. The report's `protocol.fixture`
block must read `LRV4 · return frames [59] · WR [9.8333, 9.8333] ·
probes [2.5000, 9.8333] · scalars=derived`. **If it reads `9.6000`, the
fixture was misread and the run must be discarded**, because the row sets
would have been selected against a window in which the object does not
exist.

Two of LRV3's frozen scalars (9.6 and 9.3) are pinned rather than
derived — no rule reproduces them, they were round numbers, and the
recorded experiments selected rows with them. LRV4 uses
`scalars_source: "derived"`. That difference is disclosed in every report
and is not a defect.
