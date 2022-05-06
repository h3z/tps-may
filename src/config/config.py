PROJECT_NAME = "tps-may"
ONLINE = False

RANDOM_STATE = 42

__wandb__ = {
    "project": PROJECT_NAME,
    "entity": "hzzz",
    # mkdir /wandb/PROJECT_NAME
    "dir": f"/wandb/{PROJECT_NAME}",
    "mode": "online" if ONLINE else "offline",
}


features = [f"f_{i:02d}" for i in range(31)]
