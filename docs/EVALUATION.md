# Evaluation

This page summarizes an additional evaluation of the original pitching-event detector.

The evaluation used six pitching clips that were separate from the original demo videos. The Foot Contact, Maximum External Rotation, and Ball Release detection rules were kept unchanged.

Before running the detector, I manually annotated the target events by reviewing adjacent frames. When an event could not be identified reliably at a single frame, I kept a plausible frame range instead of forcing one exact label.

## Event Definitions

- **Foot Contact:** the first frame with reliable visual evidence that the lead foot contacts the local ground surface.
- **Maximum External Rotation (MER):** the frame or frame range of maximum visible layback in the side-view video. This is used as a 2D timing reference, not as a direct measurement of shoulder external rotation.
- **Ball Release:** the first frame where the ball can clearly be seen separated from the throwing hand.
- **Visual Foot Plant:** an additional reference used when reviewing Foot Contact errors. It represents the point where the lead foot has approximately completed landing and reached a planted or settled position.

## Results

The table shows the manual annotation followed by the detector prediction (annotation → prediction).

| Clip | Foot Contact | MER | Ball Release | Observation |
|---|---|---|---|---|
| `clip_01` | 77–78 → **78** | 82–83 → **81** | 85 → **85** | MER is 1 frame before the nearest annotated boundary. |
| `clip_02` | 141–144 → **144** | 147 → **147** | 150 → **149** | Ball Release is 1 frame early. |
| `clip_03` | 103 → **103** | 109 → **109** | 111 → **112** | Ball Release is 1 frame late. |
| `clip_04` | 91–92 → **90** | 97 → **98** | 100 → **100** | Foot Contact is 1 frame early and MER is 1 frame late. Pose landmarks are less stable in this clip. |
| `clip_05` | 120–121 → **121** | 126 → **126** | 129 → **129** | All three predictions fall within the annotated frames or ranges. |
| `clip_06` | 631–632 → **643** | 652–654 → **651** | 668–669 → **670** | Foot Contact is later than strict first contact, but frame 643 falls within the annotated Visual Foot Plant range of 640–645. |

## Summary

- **Foot Contact:** 4 of 6 predictions fall inside the annotated first-contact range.
- **Maximum External Rotation:** all 6 predictions are either inside the annotated range or 1 frame from the nearest boundary.
- **Ball Release:** all 6 predictions are either inside the annotated range or 1 frame from the nearest boundary.

The main limitation appears in Foot Contact. The original heuristic can sometimes identify a planted or settled foot position rather than the first visible instant of ground contact.

## Limitations

This is a small evaluation of six clips, so it should be treated as a check of the detector's behavior rather than a large accuracy benchmark.

The annotations are based on monocular side-view video, so some events are visually ambiguous and are represented by frame ranges.

`clip_06` is slow-motion footage recorded with a training-center high-speed camera. The video file is encoded at 75 FPS, but the original capture frame rate is unknown, so its frame differences should not be converted directly into real-time milliseconds.
