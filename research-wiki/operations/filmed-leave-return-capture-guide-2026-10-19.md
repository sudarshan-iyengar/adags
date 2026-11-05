# Filmed leave-and-return capture: how to shoot it (2026-10-19)

Purpose: one bounded test (3-4 days, hard stop). On REAL filmed footage, do monocular dynamic methods with a static
background (Multi4D in its monocular configuration, Shape of Motion, MoSca) keep rendering an object after it has left
and the moving camera has seen the empty spot? If at least two of the three do, there is a method story. If not, the
leave-and-return line stops for good.

## What the footage must give us

- **Training video:** one hand-held phone moving slowly around the scene. This is the monocular input, as in DyCheck.
- **Held-out ground truth:** two static phones on tripods. Their own footage is the real ground truth during the
  absence, and the spot the object vacated comes straight from their frames (object present vs absent, same static
  camera). Nothing is authored or composited.

## Setup (once)

- **Scene:** a table with a textured surface and a textured background (books, a patterned cloth, posters). A mostly
  static object of hand size, matte and opaque: a mug, a cereal box, a toy. Avoid glass, metal, mirrors and screens.
  Keep windows out of view.
- **Light:** constant indoor light. No daylight changes, no flicker.
- **Phones:** all three in video mode, 30 fps or more, 1080p or 4K, the same lens (1x main camera, no zoom).
  - LOCK exposure, white balance and focus on every phone. On iPhone: long-press to lock AE/AF; use an app such as
    Blackmagic Camera or FiLMiC if you can, to lock white balance too.
  - Turn off stabilisation if the app allows it.
- **Static phones:** on tripods at about 1-1.5 m from the object, both looking at it.
  - One placed INSIDE the range the moving phone will sweep (an interpolation view).
  - One slightly OUTSIDE that range (an extrapolation view), about 20-30 degrees beyond the sweep's end.

## One take (about 40-50 s)

1. Start all three recordings. Do one sharp hand clap where every phone can see it (this is the sync mark).
2. **0-10 s, present.** The moving phone sweeps slowly in an arc of about 60-90 degrees around the table, at about 1 m,
   keeping the object near the centre of the frame.
3. **About 10 s, leave.** A hand reaches in, picks the object up and takes it completely out of every phone's view.
   The hand leaves the frame too.
4. **10-25 s, absent.** The moving phone keeps sweeping and LOOKS AT THE EMPTY SPOT from several angles: go back and
   forth across the arc at least once. This is what makes the absence "observed".
5. **About 25 s, return.** The hand puts the same object back in the SAME place, then leaves the frame.
6. **25-40 s, present again.** Keep sweeping.
7. Stop all recordings.

Keep the motion smooth and slow: no fast pans, no walking in the background, nobody else in view.

## Takes to shoot (about an hour in total)

- **Take A:** the protocol above with object 1 (e.g. a mug).
- **Take B:** the same with object 2 of a different size or colour (e.g. a box), in a different place on the table.
- **Take C (optional):** like A, but the object does NOT return: the absence lasts to the end.
- **Calibration sweep:** 20 s of the moving phone sweeping the empty, static scene with the tripods untouched. This
  helps solve all cameras' poses in one coordinate frame.
- Keep every take with the tripods in the SAME positions. If a tripod moves, say which take.

## Hand-over

- The original video files from all three phones, not re-encoded and not edited (AirDrop or a cable; no messaging
  apps, they compress).
- A short note: phone models, which phone was moving or static, the approximate leave and return times per take, and
  anything that went wrong.
- Upload to Leonardo or a shared drive and tell me the path. I will extract and sync the frames, solve the poses with
  COLMAP (moving-camera frames plus a few frames from each static phone), build the vacated-region masks from the static
  phones, and run the three methods under their own monocular protocols.

## What we will measure

- In each static held-out view, during the absence: the vacated region vs a ring of static surface around it, the same
  measurement as before (ghost minus ring, and the share of vacated pixels that look like the object).
- After the return: whether the object comes back cleanly.
- Decision rule:
  - **Go:** at least two of the three methods keep the object (ghost well below the ring, most vacated pixels closer to
    the object) in both takes.
  - **Stop:** otherwise.
