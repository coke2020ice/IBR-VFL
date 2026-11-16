from pathlib import Path
import runpy
if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "code/scripts/run_long_tail_ebrb_experiment.py"), run_name="__main__")
