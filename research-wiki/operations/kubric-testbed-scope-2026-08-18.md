# Kubric surround-rig testbed — SCOPE MEMO (nothing downloaded, nothing built)

Date: 2026-08-18. Status: **SCOPE ONLY.** No download, no build, no render, no
Kubric dependency added. This memo exists so that if the user authorizes the
testbed, its job is already defined; and so that if they do not, the reason it
was wanted is on record.

Reads [[dataset-admission-matrix-2026-08-18]] section 3,
[[elgs-audit-prereg-2026-08-18]] section 6,
[[elgs-absence-diagnostic-result]], [[user-decision-memo-2026-08-18]].

## 1. The specific hole this testbed fills — and it is NOT "we would like ground truth"

The amended audit preregistration concedes one thing it structurally cannot
deliver ([[elgs-audit-prereg-2026-08-18]] section 6):

> nothing here estimates the miss rate on genuine full-view absence, enclosure,
> reveal, or identity-ambiguous events, because **no window is known to be
> genuinely absent**.

Its presence decoys bound the instrument's **false-positive** side only: a
decoy is a window where the object demonstrably IS visible, so a decoy scored
A3 is a false alarm. There is no counterpart on the other side, because
producing one requires a window where the object is *known to be absent* — and
on DiVa-360 nothing knows that. The absence diagnostic found the same wall from
the other direction: 0 of 597 candidates corroborated, and the reason it cannot
convert that into a statement about the scene is that it has no access to
truth.

**So the testbed's job is one number: the instrument's sensitivity.** Not a
benchmark, not a training corpus, not a novelty demonstration. Given a scene
where absence intervals are known exactly, run the *same* frozen instrument
over it and measure what fraction of genuine full-view absences it recovers.
That single quantity is what turns every A3 and A_S count from a bound into an
estimate.

This is a sharper and much smaller specification than "build a synthetic
dataset", and it is worth stating because the larger version has no clear
stopping point.

## 2. What the scene must have, and what it does not need

**Must have:**

| requirement | why |
|---|---|
| a **surround** camera rig, >= ~20 cameras with real azimuthal diversity | the whole point of full-multiview absence is a second direction from which to rule out "occluded from here". This is exactly what disqualifies Google Immersive's sub-metre dome and N3V's frontal arc. |
| **exact per-instance visibility per camera per frame** | the ground truth the instrument's `v` flag is being compared against |
| **scripted disappearance and return** of an identified object | absence must actually occur, at known times, with a known identity across the gap |
| **enclosure and reveal** as a distinct scripted event | the case the audit's decoys cannot probe: the object is present but occluded by a container. An instrument that scores enclosure as absence is not measuring absence. |
| a hand or manipulator that **occupies the vacated site** | the standing `ACKNOWLEDGED_LIMIT_r3` confound: C2/C3 cannot separate "still there, untracked" from "left, and the hand now fills the site". Only a scripted version of this separates them. |
| foreground masks in the DiVa-360 convention | so the frozen instrument runs unmodified, which is the only way the sensitivity number transfers |

**Does NOT need:** photorealism, a large corpus, texture variety, many scenes,
long sequences, or any resemblance to a publishable dataset. A handful of short
sequences with correct geometry and correct labels is worth more than a large
realistic one, and a large one would invite exactly the misuse this memo is
trying to prevent — see section 5.

## 3. Why Kubric rather than the alternatives

* **MOVi-MC-AC** (Kubric-generated, CC BY 4.0, ~1.49 TB) already ships modal
  and amodal segmentation, amodal RGB, per-instance ids, depth and collision
  metadata from the simulator rather than from a tracker. **Wrong on two of the
  six requirements**: 6 cameras, not surround; generic clutter, not
  hand-object manipulation with scripted leave-and-return. It is a good
  demonstration that the ground truth exists and a poor substitute for the
  scene we need.
* **CMU Panoptic Studio** gives a true ~5 m geodesic dome and real humans, and
  its triangulated 3D pose could corroborate presence semi-independently of any
  single 2D tracker. But it is pose-based rather than object-identity-based, and
  its sequences are short social interactions unlikely to contain scripted
  leave-and-return. Worth keeping on the list as a *real-data* corroboration
  route; not a sensitivity testbed.
* **Kubric** is the generator behind MOVi-MC-AC and the only route to a scene
  that satisfies all six requirements. It is an engineering effort, not an
  acquisition, and that is the honest cost.

## 4. Bounded work plan, if authorized — four gates, each killing cheaply

| stage | deliverable | gate to pass before the next |
|---|---|---|
| **K1** | Kubric installed in a container; the shipped example renders; the outputs' segmentation and instance-id fields read and their exact semantics confirmed against the renderer | the example renders and the label semantics are what the docs claim |
| **K2** | ONE 60-frame scene: 24-camera surround rig, one object, one scripted full-occlusion-free disappearance (object leaves the volume) and one return; masks exported in the DiVa-360 convention | the frozen conversion accepts it and the census runs on it unmodified |
| **K3** | the frozen absence instrument run over K2 **unchanged**, and the recovered-absence fraction reported against the known intervals | a sensitivity number exists, whatever it is |
| **K4** | two further scripted variants — enclosure/reveal, and hand-occupies-vacated-site — and the same measurement on each | three sensitivity numbers, one per event class |

**K3 is the deliverable.** K1 and K2 are means; K4 is what makes the number
useful rather than a single point. If K1 or K2 fails the testbed is abandoned
having cost very little, which is the reason for staging it this way.

**Cost, unestimated and labelled as such.** Kubric renders on CPU or GPU
depending on the backend, and no measurement of either exists in this project.
Any authorization should be for **K1 only**, with K2's cost measured before it
is committed. The one thing that can be said now is that storage is not a
constraint: Apollo has 31 TiB free and a 24-camera 60-frame scene at DiVa-360
resolution is on the order of a few GiB.

## 5. The three ways this testbed could be misused, named in advance

1. **As evidence for a claim about real data.** A sensitivity number measured on
   synthetic geometry transfers to DiVa-360 only under an assumption nobody has
   tested. It bounds the instrument, not the dataset. Per
   [[dataset-admission-matrix-2026-08-18]], synthetic material is admissible for
   **C1/C2 mechanism work and never for C3**.
2. **As a substitute for the supply question.** Scripting a disappearance proves
   the instrument can see one; it says nothing about whether DiVa-360 contains
   any. The event-supply route's status is unchanged by anything a testbed
   produces.
3. **As a training corpus.** Training EL-GS on synthetic scenes and reporting
   the result would be a different project, and a weak one.

## 6. Priority

**Below** the ImViD Opera pilot and the presence-substrate viability test, and
**above** any further real-dataset acquisition — because it is the only item on
the list that addresses a hole the audit preregistration explicitly cannot
close by itself. It remains a user decision (D5) and nothing in this memo
authorizes K1.
