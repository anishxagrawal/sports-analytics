# experiments/test_entity_layer.py

"""
Sanity test script for Entity Layer.

Validates BaseEntity, Player, and EntityManager behavior with simple
manual test cases. Run directly with: python experiments/test_entity_layer.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from entities.base_entity import BaseEntity
from entities.player import Player
from entities.entity_manager import EntityManager


def test_base_entity():
    """Test BaseEntity creation, updates, and lifecycle."""
    print("=" * 60)
    print("TEST: BaseEntity")
    print("=" * 60)
    
    # Create entity
    entity = BaseEntity(track_id=1, class_id=0, missing_threshold=3)
    print(f"✓ Created: {entity}")
    
    # Update multiple times
    entity.update(bbox=(100, 200, 150, 250), frame_index=0)
    entity.update(bbox=(110, 210, 160, 260), frame_index=1)
    entity.update(bbox=(120, 220, 170, 270), frame_index=2)
    
    trajectory = entity.get_trajectory()
    print(f"✓ Trajectory length: {len(trajectory)} (expected: 3)")
    assert len(trajectory) == 3, "Trajectory length mismatch"
    
    # Check latest position
    position = entity.get_position()
    expected_x = (120 + 170) / 2.0
    expected_y = (220 + 270) / 2.0
    print(f"✓ Latest position: {position} (expected: ({expected_x}, {expected_y}))")
    assert position == (expected_x, expected_y), "Position mismatch"
    
    # Check active state
    print(f"✓ Is active: {entity.is_active()} (expected: True)")
    assert entity.is_active() == True, "Should be active"
    
    # Mark missing repeatedly
    entity.mark_missing()
    print(f"✓ Missing frames: {entity.missing_frames} (expected: 1)")
    assert entity.missing_frames == 1, "Missing frames mismatch"
    assert entity.is_active() == True, "Should still be active"
    
    entity.mark_missing()
    entity.mark_missing()
    print(f"✓ Missing frames: {entity.missing_frames} (expected: 3)")
    print(f"✓ Is active: {entity.is_active()} (expected: False)")
    assert entity.missing_frames == 3, "Missing frames mismatch"
    assert entity.is_active() == False, "Should be inactive"
    
    print("✓ BaseEntity tests passed\n")


def test_player():
    """Test Player entity behavior."""
    print("=" * 60)
    print("TEST: Player")
    print("=" * 60)
    
    # Create player
    player = Player(track_id=5, class_id=0)
    print(f"✓ Created: {player}")
    
    # Check entity type
    print(f"✓ Entity type: {player.entity_type} (expected: 'player')")
    assert player.entity_type == "player", "Entity type mismatch"
    
    # Update and verify BaseEntity behavior works
    player.update(bbox=(50, 100, 100, 200), frame_index=0)
    position = player.get_position()
    expected_x = (50 + 100) / 2.0
    expected_y = (100 + 200) / 2.0
    print(f"✓ Position after update: {position} (expected: ({expected_x}, {expected_y}))")
    assert position == (expected_x, expected_y), "Position mismatch"
    
    # Check to_dict includes entity_type
    player_dict = player.to_dict()
    print(f"✓ to_dict() includes entity_type: {player_dict.get('entity_type')} (expected: 'player')")
    assert player_dict.get('entity_type') == 'player', "to_dict missing entity_type"
    
    print("✓ Player tests passed\n")


def test_entity_manager():
    """Test EntityManager with simulated frame sequence."""
    print("=" * 60)
    print("TEST: EntityManager")
    print("=" * 60)
    
    manager = EntityManager()
    print(f"✓ Created: {manager}")
    
    # Frame 1: tracks [1, 2]
    print("\n--- Frame 1: tracks [1, 2] ---")
    tracks_frame1 = [
        {"track_id": 1, "bbox": (100, 200, 150, 250), "confidence": 0.9, "class_id": 0},
        {"track_id": 2, "bbox": (300, 200, 350, 250), "confidence": 0.85, "class_id": 0}
    ]
    manager.update(tracks_frame1, frame_index=0)
    
    all_players = manager.get_all_players()
    active_players = manager.get_active_players()
    print(f"✓ Total players: {len(all_players)} (expected: 2)")
    print(f"✓ Active players: {len(active_players)} (expected: 2)")
    assert len(all_players) == 2, "Should have 2 players"
    assert len(active_players) == 2, "Should have 2 active players"
    
    # Frame 2: tracks [1, 2]
    print("\n--- Frame 2: tracks [1, 2] ---")
    tracks_frame2 = [
        {"track_id": 1, "bbox": (110, 210, 160, 260), "confidence": 0.9, "class_id": 0},
        {"track_id": 2, "bbox": (310, 210, 360, 260), "confidence": 0.85, "class_id": 0}
    ]
    manager.update(tracks_frame2, frame_index=1)
    
    player1 = manager.get_player_by_track_id(1)
    player2 = manager.get_player_by_track_id(2)
    
    traj1_len = len(player1.get_trajectory())
    traj2_len = len(player2.get_trajectory())
    print(f"✓ Player 1 trajectory length: {traj1_len} (expected: 2)")
    print(f"✓ Player 2 trajectory length: {traj2_len} (expected: 2)")
    assert traj1_len == 2, "Player 1 should have 2 positions"
    assert traj2_len == 2, "Player 2 should have 2 positions"
    
    active_players = manager.get_active_players()
    print(f"✓ Active players: {len(active_players)} (expected: 2)")
    assert len(active_players) == 2, "Should have 2 active players"
    
    # Frame 3: tracks [2] only
    print("\n--- Frame 3: tracks [2] only ---")
    tracks_frame3 = [
        {"track_id": 2, "bbox": (320, 220, 370, 270), "confidence": 0.85, "class_id": 0}
    ]
    manager.update(tracks_frame3, frame_index=2)
    
    player1 = manager.get_player_by_track_id(1)
    player2 = manager.get_player_by_track_id(2)
    
    print(f"✓ Player 1 is active: {player1.is_active()} (expected: True)")
    print(f"✓ Player 1 missing frames: {player1.missing_frames} (expected: 1)")
    print(f"✓ Player 2 is active: {player2.is_active()} (expected: True)")
    print(f"✓ Player 2 trajectory length: {len(player2.get_trajectory())} (expected: 3)")
    
    assert player1.is_active() == True, "Player 1 should still be active (1 missing frame)"
    assert player1.missing_frames == 1, "Player 1 should have 1 missing frame"
    assert player2.is_active() == True, "Player 2 should be active"
    assert len(player2.get_trajectory()) == 3, "Player 2 should have 3 positions"
    
    active_players = manager.get_active_players()
    all_players = manager.get_all_players()
    print(f"✓ Active players: {len(active_players)} (expected: 2)")
    print(f"✓ Total players: {len(all_players)} (expected: 2)")
    assert len(active_players) == 2, "Should have 2 active players"
    assert len(all_players) == 2, "Should have 2 total players"
    
    print("✓ EntityManager tests passed\n")


def main():
    """Run all tests."""
    print("\nRunning Entity Layer Sanity Tests\n")
    
    try:
        test_base_entity()
        test_player()
        test_entity_manager()
        
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()