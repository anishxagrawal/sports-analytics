# CAMERA CALIBRATION & WORLD PROJECTION - Integration Guide

## Overview

Phase 1 of camera motion compensation is complete. This system enables:
- **Camera motion estimation** via homography (homography between frames)
- **Real-world coordinate projection** (image pixels → meters on pitch)
- **World-space velocity measurement** (m/s instead of pixels/frame)

## Architecture

```
Input Frame
    ↓
[1] Pitch Line Detection
    └─→ Extract white lines from frame
    └─→ Keypoints at line intersections
    ↓
[2] Homography Estimation (RANSAC)
    └─→ Match frame keypoints to reference frame
    └─→ Fit 3×3 homography matrix
    └─→ Validate with geometric checks
    ↓
[3] Temporal Smoothing (Buffer + EMA)
    └─→ Buffer recent homographies
    └─→ Smooth with exponential moving average
    └─→ Detect camera cuts
    ↓
[4] World Coordinate Projection
    └─→ Transform image → reference image via H
    └─→ Transform reference image → world via calibration
    └─→ Result: (X, Y) in meters
    ↓
[5] Velocity Computation
    └─→ Track position history per object
    └─→ Compute ∆position / ∆time
    └─→ Result: (vx, vy, speed) in m/s
    ↓
Output: {world_position, velocity, speed}
```

## Module Reference

### 1. **pitch_detector.py** - Line Detection
Detects white pitch lines using HSV thresholding + morphology + Hough.

**Key Classes:**
- `Line` - Detected line with endpoints, length, orientation
- `detect_pitch_lines()` - HSV-based line detection
- `extract_keypoints_from_lines()` - Get grid intersections + endpoints

**Usage:**
```python
from spatial.pitch_detector import detect_pitch_lines, extract_keypoints_from_lines

frame = cv2.imread('frame.jpg')
lines = detect_pitch_lines(frame)
keypoints = extract_keypoints_from_lines(lines, frame.shape[:2])

print(f"Found {len(lines)} lines and {len(keypoints)} keypoints")
```

### 2. **homography_estimator.py** - Homography Fitting
Estimates 3×3 homography using RANSAC with geometric validation.

**Key Classes:**
- `HomographyResult` - Result with H, confidence, inlier_ratio, errors
- `estimate_homography_ransac()` - Fit H with RANSAC + validation
- `decompose_homography()` - Extract rotation, scale, translation

**Usage:**
```python
from spatial.homography_estimator import estimate_homography_ransac

result = estimate_homography_ransac(
    src_points,  # N×2 reference keypoints
    dst_points,  # N×2 current frame keypoints
    max_reprojection_error=5.0
)

if result.is_valid:
    print(f"H confidence: {result.confidence:.2f}")
    print(f"Inlier ratio: {result.inlier_ratio:.1%}")
    H = result.H
```

### 3. **world_projector.py** - Coordinate Transformation
Projects image coordinates → world coordinates using calibrated affine/homography.

**Key Classes:**
- `PitchModel` - Pitch dimensions and structural keypoints
- `WorldProjector` - Image↔world coordinate mapping

**Usage:**
```python
from spatial.world_projector import WorldProjector

# Calibration: 4 reference points
ref_image_pts = [(100, 500), (1800, 500), (100, 100), (1800, 100)]
ref_world_pts = [(0, 0), (105, 0), (0, 68), (105, 68)]

projector = WorldProjector(ref_image_pts, ref_world_pts, method='affine')

# Per-frame projection
H = homography_matrix  # From estimator
image_point = (640, 400)  # pixels
world_point = projector.image_to_world(image_point, H)
print(f"World position: {world_point[0]:.1f}m, {world_point[1]:.1f}m")
```

### 4. **homography_buffer.py** - Temporal Smoothing
Buffers and smooths homographies with cut detection.

**Key Classes:**
- `TemporalHomographyBuffer` - EMA smoothing + cut detection
- `KalmanHomographyFilter` - (Optional) Kalman filtering alternative

**Usage:**
```python
from spatial.homography_buffer import TemporalHomographyBuffer

buffer = TemporalHomographyBuffer(window_size=5, smoothing_alpha=0.3)

for frame_idx, (H, confidence) in enumerate(homographies):
    H_smooth = buffer.add_observation(H, confidence, frame_idx)
    # Use H_smooth for projection
```

### 5. **world_velocity.py** - Velocity Estimation
Computes m/s from world-space position history.

**Key Classes:**
- `WorldVelocityEstimator` - Position history + EMA filtering
- `VelocityAccumulator` - Aggregate statistics

**Usage:**
```python
from spatial.world_velocity import WorldVelocityEstimator

vel_est = WorldVelocityEstimator(fps=30.0, entity_type='player')

for frame_idx, world_pos in enumerate(positions):
    vx, vy, speed = vel_est.update(world_pos, frame_idx, quality=0.8)
    print(f"Speed: {speed:.2f} m/s")
```

### 6. **world_projection_pipeline.py** - Full Pipeline
Integrated pipeline combining all modules.

**Key Classes:**
- `WorldProjectionPipeline` - Complete image→world pipeline

**Usage:**
```python
from spatial.world_projection_pipeline import WorldProjectionPipeline

# One-time initialization
reference_frame = cv2.imread('reference.jpg')
ref_world_pts = [(0, 0), (105, 0), (0, 68), (105, 68)]

pipeline = WorldProjectionPipeline(
    reference_frame,
    ref_world_pts,
    fps=30.0
)

# Per-frame processing
for frame_idx, frame in enumerate(video):
    detections = detector.detect(frame)
    results = pipeline.process_frame(frame, detections, frame_idx)
    
    for det in results:
        print(f"Player at {det['world_position']}, speed {det['speed']:.2f} m/s")
```

### 7. **calibration_tool.py** - Reference Frame Calibrator
Interactive tool for annotating reference frame keypoints.

**Key Classes:**
- `ReferenceFrameCalibrator` - Store correspondences + compute transforms
- `InteractiveCalibratorUI` - CLI-based calibration
- `ClickableCalibrator` - GUI-based calibration

**Usage:**
```python
# CLI calibration
from spatial.calibration_tool import InteractiveCalibratorUI

ui = InteractiveCalibratorUI()
ui.run_cli('reference.jpg', 'calibration.json')

# Then in code:
from spatial.calibration_tool import ReferenceFrameCalibrator

img_pts, world_pts = ReferenceFrameCalibrator.load_calibration('calibration.json')
```

## Integration Steps

### Step 1: Prepare Reference Frame

1. Extract first frame from video with good pitch visibility:
```python
video_reader = VideoReader('video.mp4')
reference_frame, _ = next(video_reader)
cv2.imwrite('reference.jpg', reference_frame)
```

2. Calibrate reference frame (annotate 4+ keypoints):
```bash
python -c "from spatial.calibration_tool import InteractiveCalibratorUI; \
ui = InteractiveCalibratorUI(); \
ui.run_cli('reference.jpg', 'calibration.json')"
```

3. Verify calibration was saved:
```bash
cat calibration.json
```

### Step 2: Use Pipeline in main.py

Replace existing projection pipeline with world projection:

```python
from spatial.world_projection_pipeline import WorldProjectionPipeline
from spatial.calibration_tool import ReferenceFrameCalibrator

# Load reference frame and calibration
reference_frame = cv2.imread('reference.jpg')
img_pts, world_pts = ReferenceFrameCalibrator.load_calibration('calibration.json')

# Initialize pipeline
world_pipeline = WorldProjectionPipeline(
    reference_frame,
    world_pts,
    fps=fps,
    enable_homography=True,
    enable_velocity=True
)

# Main loop
for frame_idx, frame in enumerate(video):
    detections = detector.detect(frame)
    tracks = tracker.update(detections, frame, frame_idx)
    
    # Replace old projection with world projection
    enriched_tracks = world_pipeline.process_frame(frame, tracks, frame_idx)
    
    for track in enriched_tracks:
        world_pos = track['world_position']
        speed = track['speed']
        print(f"Player: {world_pos[0]:.1f}m, {world_pos[1]:.1f}m @ {speed:.2f}m/s")
```

### Step 3: Validation & Tuning

**Check diagnostics:**
```python
diags = world_pipeline.get_diagnostics()
print(f"Frames processed: {diags['frame_count']}")
print(f"Homography buffer: {diags['homography_buffer']}")
```

**Validate calibration:**
```python
is_valid, stats = world_pipeline.world_projector.validate_calibration()
print(f"Calibration valid: {is_valid}")
print(f"Max error: {stats['max_error_m']:.2f}m")
```

## Configuration & Tuning

### Line Detection

```python
lines = detect_pitch_lines(
    frame,
    blur_kernel=5,
    morph_kernel_size=5,
    hough_threshold=50,
    min_line_length=30,
    max_line_gap=10
)
```

**Tuning:**
- `hough_threshold`: Increase if too few lines detected
- `min_line_length`: Decrease for faint lines, increase for noise reduction
- `max_line_gap`: Increase to connect broken segments

### Homography Estimation

```python
result = estimate_homography_ransac(
    src_points,
    dst_points,
    max_reprojection_error=5.0,  # Pixels
    confidence_level=0.99,
    min_inliers=4
)
```

**Tuning:**
- `max_reprojection_error`: Increase for noisy matches, decrease for precision
- `confidence_level`: RANSAC confidence (higher = more iterations)
- `min_inliers`: Minimum acceptable matches

### Temporal Smoothing

```python
buffer = TemporalHomographyBuffer(
    window_size=5,           # History length
    smoothing_alpha=0.3,     # EMA weight (0=smooth, 1=responsive)
    cut_detection_threshold=0.3  # Radians
)
```

**Tuning:**
- `window_size`: Larger = smoother but slower response
- `smoothing_alpha`: 0.2-0.5 typical (higher = more responsive)

### Velocity Filtering

```python
vel_est = WorldVelocityEstimator(
    fps=30.0,
    smoothing_alpha=0.3,
    max_history=30,
    entity_type='player'
)
```

**Tuning:**
- `smoothing_alpha`: 0.2-0.5 for less jitter
- `max_history`: Keep 20-30 frames for smooth estimation

## Failure Handling

The system is designed to fail gracefully:

1. **No pitch lines detected** → Falls back to last valid homography → Identity
2. **Poor homography fit** → Rejected (confidence < 0.3) → Uses fallback
3. **Camera cut detected** → Resets buffer → Uses identity transform
4. **Unrealistic velocity** → Clipped to max speed → Returns clipped value

All failures are logged but don't crash the pipeline.

## Performance

- **Pitch line detection**: ~5-10ms per frame
- **Homography RANSAC**: ~50-100ms per frame (depends on feature count)
- **Temporal smoothing**: <1ms
- **Coordinate projection**: <1ms per point (vectorized for batches)
- **Velocity computation**: <1ms per object

**Optimization:**
- Run line detection every N frames (e.g., every 5 frames)
- Use batch projection for many objects
- Consider GPU acceleration for RANSAC if needed

## Testing

### Test 1: Line Detection
```python
from spatial.pitch_detector import detect_pitch_lines, visualize_pitch_lines

frame = cv2.imread('reference.jpg')
lines = detect_pitch_lines(frame)
vis = visualize_pitch_lines(frame, lines)
cv2.imshow('Lines', vis)
cv2.waitKey(0)
```

### Test 2: Homography
```python
from spatial.homography_estimator import estimate_homography_ransac

# Generate synthetic test case
src = np.array([[0, 0], [100, 0], [0, 100], [100, 100]], dtype=np.float32)
dst = np.array([[10, 10], [110, 5], [5, 105], [105, 105]], dtype=np.float32)

result = estimate_homography_ransac(src, dst)
print(f"Is valid: {result.is_valid}, Confidence: {result.confidence:.2f}")
```

### Test 3: End-to-End
```python
from spatial.world_projection_pipeline import WorldProjectionPipeline

# Use sample calibration
from spatial.calibration_tool import create_sample_calibration
create_sample_calibration('/tmp/sample_cal.json')

# Load and test
img_pts, world_pts = ReferenceFrameCalibrator.load_calibration('/tmp/sample_cal.json')
pipeline = WorldProjectionPipeline(reference_frame, world_pts, fps=30.0)

# Process test frame
detections = [{'bbox': (100, 500, 110, 510), 'class_id': 0, 'track_id': 1}]
results = pipeline.process_frame(frame, detections, 0)
print(f"World position: {results[0]['world_position']}")
```

## Next Steps (Phase 2+)

1. **Deep feature matching** - Use SIFT/ORB for generic features
2. **Zoom compensation** - Handle moderate zoom changes
3. **Kalman filtering** - Replace EMA with more sophisticated filter
4. **Multi-view calibration** - Use multiple reference frames
5. **Automatic reference frame selection** - Auto-pick best frame
6. **GPU acceleration** - Optimize line detection and RANSAC

## Troubleshooting

**Problem: Very few pitch lines detected**
- Solution: Check lighting conditions, adjust `hough_threshold` down, verify frame quality

**Problem: High reprojection error**
- Solution: Increase `max_reprojection_error` in calibration, check reference frame alignment

**Problem: Velocity spikes/jitter**
- Solution: Increase `smoothing_alpha`, increase position history window

**Problem: World coordinates outside [0, 105] × [0, 68]**
- Solution: Check calibration keypoints, validate reference frame annotations

## References

- OpenCV Homography: https://docs.opencv.org/master/d9/d0c/group__calib3d.html
- RANSAC: Fischler & Bolles (1981)
- Pitch line detection: Hough transform
- EMA: Simple temporal smoothing technique

---

**Author Notes:**
This implementation prioritizes **interpretability** and **correctness** over speed. Every stage is validated and can fail gracefully. Use the logging output to debug issues. All assumptions are explicit (planar pitch, known dimensions, single camera).
