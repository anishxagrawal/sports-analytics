# src/main.py

"""
Main orchestration script for sports analytics pipeline.

Connects all pipeline components and runs frame-by-frame processing
on video input with commentary generation.
"""

from src.core.video import VideoReader
from src.core.detector import YOLODetector
from src.core.tracker import Tracker
from src.entities.entity_manager import EntityManager
from src.analytics.events import detect_player_events
from src.commentary.engine import CommentaryEngine
from src.commentary.prompt_builder import PromptBuilder
from src.commentary.llm_adapter import LLMAdapter


def main():
    """Run the sports analytics pipeline on a video file."""
    
    # Configuration
    video_path = "data/videos/test_video.mp4"
    model_path = "yolov8n.pt"
    conf_threshold = 0.5
    allowed_classes = {0}  # Person class
    
    # Initialize components
    print("Initializing pipeline...")
    
    video_reader = VideoReader(video_path)
    fps = video_reader.fps
    
    detector = YOLODetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        allowed_classes=allowed_classes,
        device="cpu"
    )
    
    tracker = Tracker(frame_rate=fps)
    
    entity_manager = EntityManager()
    
    commentary_engine = CommentaryEngine(cooldown_seconds=5.0)
    prompt_builder = PromptBuilder()
    llm_adapter = LLMAdapter()
    
    print(f"Processing video at {fps} FPS...\n")
    
    # Frame-by-frame processing loop
    for frame, metadata in video_reader:
        frame_idx = metadata['frame_idx']
        timestamp = metadata['timestamp']
        
        # Detection
        detections = detector.detect(frame)
        
        # Tracking
        tracks = tracker.update(detections, frame)
        
        # Entity management
        entity_manager.update(tracks, frame_idx, timestamp)
        
        # Event detection
        active_players = entity_manager.get_active_players()
        all_events = []
        
        for player in active_players:
            player_events = detect_player_events(player, fps)
            all_events.extend(player_events)
        
        # Commentary pipeline
        if len(all_events) > 0:
            # Convert events to narrative intents
            intents = commentary_engine.process_events(all_events, timestamp)
            
            if intents is not None:
                # Build LLM prompt
                prompt = prompt_builder.build(intents)
                
                if prompt is not None:
                    # Generate commentary
                    commentary = llm_adapter.generate(prompt)
                    
                    if commentary is not None:
                        print(f"[{timestamp:.2f}s] {commentary}")
    
    # Cleanup
    video_reader.release()
    print("\nProcessing complete.")


if __name__ == "__main__":
    main()