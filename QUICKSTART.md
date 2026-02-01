# Quick Start: World Projection & Velocity

## TL;DR

This system lets you measure player/ball speed in **m/s** instead of pixels/frame by:
1. Detecting pitch lines
2. Estimating camera motion (homography)
3. Projecting pixels to world coordinates (meters)
4. Computing real-world velocity

## 60-Second Setup

### 1. Calibrate Reference Frame (5 min)

Extract first good frame from your video:
```bash
python -c "
import cv2
from src.core.video import VideoReader

reader = VideoReader('data/inputs/test_video_6.mp4')
frame, _ = next(reader)
cv2.imwrite('/tmp/ref.jpg', frame)
"
```

Annotate it (CLI):
```bash
python -c "
from src.spatial.calibration_tool import InteractiveCalibratorUI
ui = InteractiveCalibratorUI()
ui.run_cli('/tmp/ref.jpg', '/tmp/calibration.json')
"
```

Follow the prompts:
- Enter 4+ pitch corner points
- Format: `x y X Y` (pixel coords, world coords in meters)
- Example: `100 500 0 0` (pixel 100,500 is world 0,0)

### 2. Use in Pipeline (2 min)

```python
from src.spatial.world_projection_pipeline import WorldProjectionPipeline
from src.spatial.calibration_tool import ReferenceFrameCalibrator
import cv2

# Load calibration
img_pts, world_pts = ReferenceFrameCalibrator.load_calibration('/tmp/calibration.json')

# Load reference frame
ref_frame = cv2.imread('/tmp/ref.jpg')

# Initialize pipeline
pipeline = WorldProjectionPipeline(
    ref_frame,
    world_pts,
    fps=30.0,
    enable_homography=True,
    enable_velocity=True
)

# Process video
for frame_idx, frame in enumerate(video):
    detections = detector.detect(frame)  # Your detections
    results = pipeline.process_frame(frame, detections, frame_idx)
    
    for detection in results:
        x, y = detection['world_position']
        vx, vy = detection['velocity']
        speed = detection['speed']
        
        print(f"Player at ({x:.1f}m, {y:.1f}m), speed: {speed:.2f} m/s")
```

## What You Get

```
Input Detection (pixels): bbox = (640, 360, 660, 380)
                    ↓
Output: {
    'world_position': (52.5, 34.0),      # meters on pitch
    'velocity': (2.3, -1.1),              # m/s (horizontal, vertical)
    'speed': 2.5,                         # m/s (magnitude)
    'homography_confidence': 0.85,        # quality of estimate
}
```

## Typical Values

| Metric | Range | Notes |
|--------|-------|-------|
| World X | 0-105m | Pitch length |
| World Y | 0-68m | Pitch width |
| Speed | 0-12 m/s | Players typically 6-10 m/s |
| Ball speed | 0-40 m/s | Passes 10-20 m/s, shots 20+ m/s |

## Troubleshooting

**"Only 2 keypoints found"**
→ Poor lighting or line visibility. Try different reference frame or adjust line detection thresholds.

**"World coordinates outside bounds"**
→ Calibration error. Re-run calibration tool and verify keypoint entries.

**"Velocity spikes"**
→ Increase EMA smoothing: `smoothing_alpha=0.2` (default 0.3)

**"Homography confidence low"**
→ More lines needed or poor match quality. Check frame quality.

## Configuration

Tune in `WorldProjectionPipeline.__init__()`:

```python
pipeline = WorldProjectionPipeline(
    ref_frame,
    world_pts,
    fps=30.0,
    enable_homography=True,      # Disable for normalized coords only
    enable_velocity=True          # Disable to skip velocity computation
)
```

Fine-tune smoothing:

```python
from src.spatial.homography_buffer import TemporalHomographyBuffer

buffer = TemporalHomographyBuffer(
    window_size=5,          # Larger = smoother
    smoothing_alpha=0.3,    # 0.2-0.5 typical
    cut_detection_threshold=0.3  # Radians
)
```

## Performance

- Per-frame latency: ~100-120ms
- Can run on 10 FPS (suitable for real-time analysis)
- GPU acceleration possible but not required

## Next Steps

1. **Test on sample video** - Process 100 frames and verify output ranges
2. **Validate calibration** - Compare world coords to ground truth
3. **Tune parameters** - Adjust based on your video conditions
4. **Integrate into analytics** - Use speed/position for events, commentary, etc.

## Code Examples

### Example 1: Extract Player Speed

```python
for detection in results:
    if detection['class_id'] == 0:  # Player
        speed = detection['speed']
        if speed > 8.0:
            print(f"Fast player! {speed:.1f} m/s")
```

### Example 2: Track Player Acceleration

```python
velocity_est = pipeline.get_velocity_estimator(track_id)
if velocity_est:
    stats = velocity_est.get_statistics()
    print(f"Avg speed: {stats['avg_speed']:.2f} m/s")
    print(f"Max speed: {stats['max_speed']:.2f} m/s")
```

### Example 3: Validate Calibration

```python
is_valid, stats = pipeline.world_projector.validate_calibration()
if is_valid:
    print(f"✓ Calibration OK (max error: {stats['max_error_m']:.2f}m)")
else:
    print(f"✗ Calibration failed (max error: {stats['max_error_m']:.2f}m)")
```

### Example 4: Get Diagnostics

```python
diags = pipeline.get_diagnostics()
print(f"Frames: {diags['frame_count']}")
print(f"H buffer: {diags['homography_buffer']}")
print(f"Velocity estimators: {diags['velocity_estimators']}")
```

## Files You'll Need

- **Reference frame**: `reference.jpg` (first frame with good line visibility)
- **Calibration**: `calibration.json` (from calibration tool)
- **Modules**: Already in `src/spatial/`:
  - `pitch_detector.py`
  - `homography_estimator.py`
  - `world_projector.py`
  - `homography_buffer.py`
  - `world_velocity.py`
  - `world_projection_pipeline.py`
  - `calibration_tool.py`

## FAQ

**Q: Do I need GPU?**
A: No. ~100ms per frame on CPU is typical.

**Q: What if pitch lines aren't visible?**
A: System falls back to last valid homography. If persistent, revert to normalized coordinates.

**Q: Can I use different camera angles?**
A: Need separate calibration per camera. Or disable homography and use normalized coords.

**Q: How accurate are the velocities?**
A: Depends on calibration quality. Typically ±0.5 m/s with good calibration.

**Q: What about zoom?**
A: Handled gracefully but not perfectly. Avoid extreme zoom for best results.

## References

- Main guide: `WORLD_PROJECTION_GUIDE.md`
- Implementation summary: `PHASE1_SUMMARY.md`
- Module docs: Each `.py` file has detailed docstrings

---

**Ready?** Run the calibration tool and start measuring real speeds!
