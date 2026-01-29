# Sports Analytics Project - File Structure

```
sports-analytics/
(venv) anish@LAPTOP-FN4ANMM5:~/cv-workspace/sports-analytics$ tree -L 4
.
FILE_STRUCTURE.md
├── README.md
├── data
│   ├── inputs
│   │   ├── football.mp4:Zone.Identifier
│   │   ├── test_video.mp4
│   │   ├── test_video_2.mp4
│   │   ├── test_video_3.mp4
│   │   ├── test_video_4.mp4
│   │   ├── test_video_5.mp4
│   │   ├── test_video_6.mp4
│   │   └── test_video_6.mp4:Zone.Identifier
│   └── videos
│       └── match.mp4
├── models
│   ├── finetuned
│   │   └── README.md
│   └── pretrained
├── requirements.txt
├── scripts
│   ├── deploy_to_roboflow.py
│   ├── extract_frames.py
│   └── split_dataset.py
└── src
    ├── analytics
    │   ├── events.py
    │   ├── motion.py
    │   └── teams
    │       └── assigner.py
    ├── commentary
    │   ├── engine.py
    │   ├── llm_adapter.py
    │   ├── memory.py
    │   └── prompt_builder.py
    ├── config
    │   └── models.py
    ├── core
    │   ├── ball_tracker.py
    │   ├── detector.py
    │   ├── smoothing.py
    │   ├── tracker.py
    │   └── video.py
    ├── entities
    │   ├── ball.py
    │   ├── base_entity.py
    │   ├── entity_manager.py
    │   ├── player.py
    │   └── referee.py
    ├── experiments
    │   └── test_entity_layer.py
    ├── main.py
    └── spatial
        ├── field_lines.py
        ├── ground_point.py
        ├── image_to_field.py
        ├── projection_pipeline.py
        └── soft_anchor.py

17 directories, 41 files
```

## Module Overview

### Core Modules (`src/core/`)
- **detector.py**: Implements YOLOv8-based object detection for players and ball
- **tracker.py**: Handles multi-object tracking and identity persistence across frames
- **ball_tracker.py**: Specialized tracker for ball movement with prediction and smoothing
- **smoothing.py**: Applies exponential moving average for stabilizing position data
- **video.py**: Video I/O operations and frame extraction utilities

### Entity System (`src/entities/`)
- **base_entity.py**: Abstract base class providing common entity interface
- **entity_manager.py**: Manages entity lifecycle, creation, and state updates
- **player.py**: Player entity with team assignment and position tracking
- **ball.py**: Ball entity representing football state and trajectory

### Spatial Utilities (`src/spatial/`)
- **ground_point.py**: Converts bounding box coordinates to ground plane contact points

### Analytics (`src/analytics/`)
- **events.py**: Detects and tracks sports-specific events (passes, shots, etc.)
- **motion.py**: Analyzes player movement patterns and motion metrics
- **teams/assigner.py**: Extracts jersey color features for team identification

### Commentary (`src/commentary/`)
- **engine.py**: Orchestrates commentary generation workflow
- **llm_adapter.py**: Interfaces with LLM APIs (OpenAI, Claude, etc.)
- **memory.py**: Maintains conversation history and match context
- **prompt_builder.py**: Constructs dynamic prompts from match state and events

### Data
- **inputs/**: Raw video files for processing
- **outputs/**: Generated output videos with annotations
