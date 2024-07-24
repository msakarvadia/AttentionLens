from pathlib import Path
from pprint import pprint


CKPT_DIR = Path("./checkpoint2/gpt2/ckpt_10/")
LAYER_NUM = 10

def load_last_checkpoint() -> Path:
    print("Searching for existing checkpoints.")
    most_recent_file = None

    if CKPT_DIR.exists():
        layer_id = f"layer-{LAYER_NUM}"
        checkpoints = filter(lambda fn: layer_id in fn.stem, list(CKPT_DIR.glob("*.ckpt")))
        checkpoints = list(checkpoints)
        checkpoints_by_ctime = {
            ckpt: ckpt.stat().st_ctime for ckpt in checkpoints
        }
        print(checkpoints)
        if checkpoints:
            most_recent_file = max(checkpoints, key=lambda ckpt: ckpt.stat().st_ctime)
        else:
            raise ValueError
    else:
        raise ValueError

    pprint(checkpoints_by_ctime)
    return most_recent_file


def main(path: Path):
    if path is None:
        path = load_last_checkpoint()
    print(f"{path=}")


if __name__ == "__main__":
    main(None)