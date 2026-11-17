# Limitations

## What the output means
"High" means the video shows a configuration and behaviour pattern (an MTW passing close to a detected pothole with a lateral shift, near a medium or large pothole)
consistent with a hazardous encounter. It does not mean an accident would happen, and no crash data, rider intent, road friction or real-unit speed is available.

## Geometry
- Monocular and relative: dy, dx, proximity and swerve are in image-derived "MTW widths/heights", not metres or km/h.
- Flat-road assumption: slopes, speed breakers and curves break the row-to-depth relation; discard encounters on visibly sloped roads.
- Perspective: without the optional correction an off-axis pothole at another depth creates a spurious lateral drift (see docs/methodology.md for the measured effect on synthetic scenes); the correction needs a horizon estimate and can hurt if it is wrong.
- Contact point = bottom-centre of the box: wrong for turning, partly hidden or truncated MTWs.
- Deceleration is confounded by the camera's own speed, so its default weight is 0 in moving mode.

## Ego-motion
- Forward motion of a dashcam is not an affine image motion; the estimator uses a mid-image band and may fall back to identity (flagged). It compensates shake and small rotations, not parallax.
- Last-known potholes (hidden by the MTW itself) are held in image space and mapped with shake only, so they drift over about `pothole_memory_seconds` of forward motion.

## Detection and tracking
- ID switches under occlusion in dense traffic; tracks shorter than `min_reliable_seconds` are flagged unreliable; bounding-box jitter can create false swerve (hence smoothing, windows and minimum history).
- COCO "motorcycle" is assumed to include scooters: verify on your footage. Riders on bicycles or three-wheelers may be mislabelled.
- Shadows, patches, water, manholes and speed breakers produce pothole false positives; far potholes are small and missed.
- Domain gap: most public pothole images are close-up road-surface photos, not potholes in traffic video at distance. Expect Test A and Test B to be far below validation; that is a finding.

## Data and evaluation
- No public dataset with potholes AND motorcycles together was found, so the combined pipeline is evaluated on a small number of curated public clips: small samples, noisy statistics, possibly fewer than 30 encounters (case-study mode).
- Test B and the tracking ground truth are annotated by the team; their agreement is checked on only 20 frames.
- Third-party videos are subject to copyright and platform terms; they are not redistributed.
- Licences and class definitions of the datasets are TO VERIFY (docs/data_sources.md).

## What the tests do and do not show
- The synthetic scenes (src/ptr/synthetic.py) have made-up geometry. Passing tests prove the code implements the stated conventions, not that they hold on real roads.
- The mock mode feeds ground-truth boxes (plus noise) into the pipeline; it does not test any detector.
