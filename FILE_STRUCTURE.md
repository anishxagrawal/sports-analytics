# Sports Analytics Project - File Structure

```
sports-analytics/
(venv) anish@LAPTOP-FN4ANMM5:~/cv-workspace/sports-analytics$ tree -L 4
.
├── FILE_STRUCTURE.md
├── README.md
├── data
│   ├── inputs
│   │   ├── test_video.mp4
│   │   ├── test_video_2.mp4
│   │   └── test_video_3.mp4
│   ├── outputs
│   │   └── output.mp4
│   └── videos
│       └── match.mp4
├── models
│   ├── finetuned
│   │   ├── README.md
│   │   ├── yolov8n_v1.pt
│   │   ├── yolov8n_v2.pt
│   │   └── yolov8n_v3.pt
│   └── pretrained
│       └── yolov8n.pt
├── requirements.txt
├── runs
│   └── detect
│       └── ball_player_v23
│           ├── BoxF1_curve.png
│           ├── BoxPR_curve.png
│           ├── BoxP_curve.png
│           ├── BoxR_curve.png
│           ├── args.yaml
│           ├── confusion_matrix.png
│           ├── confusion_matrix_normalized.png
│           ├── labels.jpg
│           ├── results.csv
│           ├── results.png
│           ├── train_batch0.jpg
│           ├── train_batch1.jpg
│           ├── train_batch100.jpg
│           ├── train_batch101.jpg
│           ├── train_batch102.jpg
│           ├── train_batch2.jpg
│           ├── val_batch0_labels.jpg
│           ├── val_batch0_pred.jpg
│           └── weights
├── scripts
│   ├── deploy_to_roboflow.py
│   ├── extract_frames.py
│   └── split_dataset.py
├── src
│   ├── __pycache__
│   │   └── main.cpython-312.pyc
│   ├── analytics
│   │   ├── __pycache__
│   │   │   ├── events.cpython-312.pyc
│   │   │   ├── motion.cpython-312.pyc
│   │   │   └── smoothing.cpython-312.pyc
│   │   ├── events.py
│   │   ├── motion.py
│   │   └── teams
│   │       ├── __pycache__
│   │       └── assigner.py
│   ├── commentary
│   │   ├── __pycache__
│   │   │   ├── engine.cpython-312.pyc
│   │   │   ├── llm_adapter.cpython-312.pyc
│   │   │   ├── memory.cpython-312.pyc
│   │   │   └── prompt_builder.cpython-312.pyc
│   │   ├── engine.py
│   │   ├── llm_adapter.py
│   │   ├── memory.py
│   │   └── prompt_builder.py
│   ├── config
│   │   ├── __pycache__
│   │   │   └── models.cpython-312.pyc
│   │   └── models.py
│   ├── core
│   │   ├── __pycache__
│   │   │   ├── ball_tracker.cpython-312.pyc
│   │   │   ├── detector.cpython-312.pyc
│   │   │   ├── smoothing.cpython-312.pyc
│   │   │   ├── tracker.cpython-312.pyc
│   │   │   └── video.cpython-312.pyc
│   │   ├── ball_tracker.py
│   │   ├── detector.py
│   │   ├── smoothing.py
│   │   ├── tracker.py
│   │   └── video.py
│   ├── entities
│   │   ├── __pycache__
│   │   │   ├── ball.cpython-312.pyc
│   │   │   ├── base_entity.cpython-312.pyc
│   │   │   ├── entity_manager.cpython-312.pyc
│   │   │   └── player.cpython-312.pyc
│   │   ├── ball.py
│   │   ├── base_entity.py
│   │   ├── entity_manager.py
│   │   └── player.py
│   ├── experiments
│   │   └── test_entity_layer.py
│   ├── main.py
│   ├── spatial
│   │   ├── __pycache__
│   │   │   └── ground_point.cpython-312.pyc
│   │   └── ground_point.py
│   └── yolov8n.pt
├── tmp
│   ├── raw_frames
│   └── roboflow_model
│       ├── model_artifacts.json
│       ├── roboflow_deploy.zip
│       ├── state_dict.pt
│       └── yolov8_ball_v1.pt
├── training
│   └── yolo
│       └── ball_detection
│           ├── README.md
│           ├── data
│           ├── data_raw
│           ├── dataset.yaml
│           ├── train.py
│           └── validate.py
├── venv
│   ├── bin
│   │   ├── Activate.ps1
│   │   ├── activate
│   │   ├── activate.csh
│   │   ├── activate.fish
│   │   ├── boxmot
│   │   ├── dotenv
│   │   ├── f2py
│   │   ├── filetype
│   │   ├── fonttools
│   │   ├── ftfy
│   │   ├── gdown
│   │   ├── isympy
│   │   ├── normalizer
│   │   ├── numpy-config
│   │   ├── pip
│   │   ├── pip3
│   │   ├── pip3.12
│   │   ├── proton
│   │   ├── proton-viewer
│   │   ├── pyftmerge
│   │   ├── pyftsubset
│   │   ├── python -> python3
│   │   ├── python3 -> /usr/bin/python3
│   │   ├── python3.12 -> python3
│   │   ├── roboflow
│   │   ├── torchfrtrace
│   │   ├── torchrun
│   │   ├── tqdm
│   │   ├── ttx
│   │   ├── ultralytics
│   │   └── yolo
│   ├── include
│   │   └── python3.12
│   ├── lib
│   │   └── python3.12
│   │       └── site-packages
│   ├── lib64 -> lib
│   ├── pyvenv.cfg
│   └── share
│       └── man
│           └── man1
├── yolo11n.pt
└── yolov8n.pt

49 directories, 116 files
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
