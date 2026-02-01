# Spatial Module: Camera Calibration & World Projection

This module provides camera motion compensation and real-world coordinate projection for sports analytics.

## Modules Overview

```
spatial/
├── pitch_detector.py              # Line detection & keypoint extraction
├── homography_estimator.py        # RANSAC homography fitting with validation
├── world_projector.py             # Image ↔ World coordinate transformation
├── homography_buffer.py           # Temporal smoothing + cut detection
├── world_velocity.py              # Real-world velocity estimation
├── world_projection_pipeline.py   # Complete end-to-end pipeline
├── calibration_tool.py            # Reference frame calibration UI
├── soft_anchor.py                 # Soft spatial anchoring (legacy)
├── ground_point.py                # Bbox to ground point conversion
├── image_to_field.py              # Normalized field space projection
├── field_lines.py                 # Field line detection
├── projection_pipeline.py          # Base projection pipeline (soft anchoring)
├── field_lines.py                 # Field line detection utilities
└── README.md                       (this file)
```

## Pipeline Architecture

### Level 1: Simple Normalized Projection (existing)
```
BBox → Ground Point → Normalized Field Space [0,1]×[0,1]
                                    ↓
                            Soft Spatial Anchoring (optional)
                                    ↓
                            Stable field-space positions
```

**Use when:**
- Camera is mostly static
- Only relative player positions matter
- Speed measurement not needed

**Modules:** `ground_point.py`, `image_to_field.py`, `soft_anchor.py`, `projection_pipeline.py`

---

### Level 2: World Coordinates with Homography (new)
```
Frame → Pitch Lines → Homography (H) → Temporal Buffer
           ↓                                    ↓
        Keypoints                        Stabilized H
                                              ↓
BBox → Ground Point → Project via H → World Coordinates (meters)
                                              ↓
                                    Position History
                                              ↓
                                         Velocity (m/s)
```

**Use when:**
- Need real-world measurements (m/s)
- Camera moves significantly
- Accurate speed analysis required

**Modules:** `pitch_detector.py`, `homography_estimator.py`, `homography_buffer.py`, `world_projector.py`, `world_velocity.py`, `world_projection_pipeline.py`, `calibration_tool.py`

## Quick Reference

### For Normalized Coordinates (existing)
```python
from spatial.projection_pipeline import ProjectionPipeline

pipeline = ProjectionPipeline(enable_anchoring=True)
results = pipeline.process_frame(detections, frame_shape)

# Results include: 'field_position', 'field_position_anchored'
# Values in [0,1] × [0,1] representing normalized pitch space
```

### For World Coordinates (new)
```python
from spatial.world_projection_pipeline import WorldProjectionPipeline

pipeline = WorldProjectionPipeline(
    reference_frame,
    world_keypoints,
    fps=30.0
)
results = pipeline.process_frame(frame, detections, frame_idx)

# Results include: 'world_position' (X,Y in meters), 'velocity' (vx,vy m/s), 'speed' (m/s)
```

## Module Dependencies

```
pitch_detector.py
    ↓
    ├─→ homography_estimator.py
    │       ↓
    │       └─→ world_projector.py
    │               ↓
    │               └─→ world_velocity.py
    │                       ↓
    │                       └─→ world_projection_pipeline.py
    │
    └─→ calibration_tool.py
            ↓
            └─→ world_projector.py
```

All modules are independent (can import individually) but compose in pipeline.

## Data Flow

### Reference Frame (One-time Setup)
```
reference_frame.jpg
    ↓
[detect_pitch_lines]      → List[Line]
    ↓
[extract_keypoints]       → List[(x,y)] in pixels
    ↓
[calibration_tool]        → User annotates world coords
    ↓
[WorldProjector init]     → Stores image↔world mapping
```

### Per-Frame Processing
```
frame[t]
    ↓
[detect_pitch_lines]      → List[Line]
    ↓
[extract_keypoints]       → List[(x,y)] in pixels
    ↓
[estimate_homography]     → H (3×3 matrix)
    ↓
[validate_homography]     → confidence, is_valid
    ↓
[temporal_buffer]         → H_stabilized
    ↓
[detect_object]           → bbox
    ↓
[world_projector]         → (X,Y) in meters
    ↓
[velocity_estimator]      → (vx, vy, speed)
```

## Configuration

### Line Detection
```python
lines = detect_pitch_lines(
    frame,
    hough_threshold=50,      # Lower = more lines
    min_line_length=30,       # Pixels
    max_line_gap=10          # Pixels
)
```

### Homography
```python
result = estimate_homography_ransac(
    src_points, dst_points,
    max_reprojection_error=5.0,  # Pixels
    confidence_level=0.99,        # RANSAC confidence
    min_inliers=4
)
```

### Temporal Smoothing
```python
buffer = TemporalHomographyBuffer(
    window_size=5,               # History length
    smoothing_alpha=0.3,         # EMA: 0=smooth, 1=responsive
    cut_detection_threshold=0.3  # Radians
)
```

### Velocity
```python
vel = WorldVelocityEstimator(
    fps=30.0,
    smoothing_alpha=0.3,         # EMA smoothing
    max_history=30,              # Frames to keep
    entity_type='player'         # 'player' or 'ball'
)
```

## Performance

| Operation | Time | Tuning |
|-----------|------|--------|
| Line detection | 5-10ms | Every N frames |
| RANSAC | 50-100ms | Fewer features |
| Temporal smooth | <1ms | - |
| Projection | <1ms | Batch operations |
| Velocity | <1ms | - |

## Testing

### Test 1: Line Detection
```python
from spatial.pitch_detector import detect_pitch_lines, visualize_pitch_lines

lines = detect_pitch_lines(frame)
vis = visualize_pitch_lines(frame, lines)
cv2.imshow('Lines', vis)
cv2.waitKey(0)
```

### Test 2: Homography
```python
from spatial.homography_estimator import estimate_homography_ransac

result = estimate_homography_ransac(src_pts, dst_pts)
assert result.is_valid
assert result.confidence > 0.5
assert result.inlier_ratio > 0.6
```

### Test 3: World Projection
```python
from spatial.world_projector import WorldProjector

projector = WorldProjector(ref_img_pts, ref_world_pts)
world_pt = projector.image_to_world((640, 360), H)

assert 0 <= world_pt[0] <= 105  # Pitch length
assert 0 <= world_pt[1] <= 68   # Pitch width
```

### Test 4: Velocity
```python
from spatial.world_velocity import WorldVelocityEstimator

vel_est = WorldVelocityEstimator(fps=30.0, entity_type='player')
vx, vy, speed = vel_est.update((52.5, 34.0), 0)

assert speed < 20.0  # Reasonable max
```

## Failure Modes

| Failure | Detection | Fallback |
|---------|-----------|----------|
| No lines detected | len(lines) == 0 | Use identity H |
| Poor homography | confidence < 0.3 | Use last valid H |
| Camera cut | rotation_angle > threshold | Reset buffer |
| Unrealistic speed | speed > MAX_SPEED | Clip to max |
| Invalid calibration | max_error > tolerance | Return None |

All failures are logged but non-fatal.

## Integration with Main Pipeline

### Minimal Change (drop-in replacement)
```python
# OLD:
enriched = projection_pipeline.process_frame(detections, frame_shape)

# NEW:
enriched = world_pipeline.process_frame(frame, detections, frame_idx)
```

### Full Usage
```python
# Initialize once
from spatial.world_projection_pipeline import WorldProjectionPipeline

pipeline = WorldProjectionPipeline(
    reference_frame=cv2.imread('ref.jpg'),
    reference_keypoints_world=[(0,0), (105,0), (0,68), (105,68)],
    fps=30.0,
    enable_homography=True,
    enable_velocity=True
)

# Per frame
for frame_idx, frame in enumerate(video):
    detections = detector.detect(frame)
    results = pipeline.process_frame(frame, detections, frame_idx)
    
    for r in results:
        print(f"Position: {r['world_position']}, Speed: {r['speed']} m/s")
```

## Troubleshooting

### Few/No Lines Detected
- Check frame lighting conditions
- Lower `hough_threshold` in line detection
- Verify pitch has visible white lines
- Try different reference frame

### High Homography Error
- Increase `max_reprojection_error` in RANSAC
- Check feature matching quality
- Verify reference frame annotations

### Velocity Spikes
- Increase `smoothing_alpha` (e.g., 0.2 instead of 0.3)
- Check position history length
- Verify detections are stable

### World Coordinates Out of Bounds
- Re-run calibration tool
- Verify reference keypoint annotations
- Check that H matrix is valid

## Future Improvements

1. **Deep feature matching** - Generic feature descriptor instead of lines
2. **GPU acceleration** - RANSAC on GPU
3. **Zoom handling** - Better compensation for zoom changes
4. **Multi-reference** - Support multiple camera angles
5. **Kalman filtering** - Replace EMA with Kalman

## References

- **Homography:** OpenCV docs, Hartley & Zisserman
- **RANSAC:** Fischler & Bolles (1981)
- **Pitch detection:** Hough transform + morphology
- **Temporal smoothing:** EMA (exponential moving average)

---

**Documentation:** See `QUICKSTART.md`, `WORLD_PROJECTION_GUIDE.md`, `PHASE1_SUMMARY.md`
