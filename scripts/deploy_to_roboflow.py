from roboflow import Roboflow
from pathlib import Path

# -------- CONFIG --------
API_KEY = "YOUR_API_KEY"            # ← replace
WORKSPACE_NAME = "anish-zgsah"       # ← replace
PROJECT_IDS = ["football-fdgwl"]     # ← replace
MODEL_NAME = "ball-yolov8-v1"        # ← must contain letters
# ------------------------

rf = Roboflow(api_key="lRVv0Y1NRbG8J5ydYj6N")
workspace = rf.workspace("anish-zgsah")

model_dir = Path("tmp/roboflow_model")

workspace.deploy_model(
    model_type="yolov8",
    model_path=str(model_dir),
    project_ids=PROJECT_IDS,
    model_name=MODEL_NAME,
    filename="yolov8_ball_v1.pt"
)

print("✅ Model deployed to Roboflow successfully")
