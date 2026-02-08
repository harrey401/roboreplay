"""RoboReplay — The DVR for robot behavior.

Record, replay, diagnose, and share robot execution data.
Framework-agnostic. Works with MuJoCo, Isaac Sim, ROS2, Gymnasium, real hardware,
or anything with a Python API.

Quick start:
    from roboreplay import Recorder, Replay, compare, export_csv, wrap

    # Record
    with Recorder("my_experiment", metadata={"robot": "panda"}) as rec:
        for step in range(100):
            rec.step(state=get_state(), action=get_action(), reward=get_reward())

    # Replay
    r = Replay("my_experiment.rrp")
    print(r)            # Summary
    print(r[50])         # Data at step 50
    print(r.channels)    # Channel names

    # Diagnose
    from roboreplay.diagnose import diagnose
    result = diagnose("my_experiment.rrp")

    # Export
    export_csv("my_experiment.rrp", output_dir="exports/")
    from roboreplay.export import export_html
    export_html("my_experiment.rrp")

    # Gymnasium
    env = wrap(gym.make("CartPole-v1"), name="cartpole")
"""

__version__ = "0.1.0"

from roboreplay.compare import compare
from roboreplay.export.csv import export_csv
from roboreplay.recorder import Recorder
from roboreplay.replay import Replay


def wrap(*args, **kwargs):
    """Lazy import for Gymnasium wrapper to avoid hard dependency."""
    from roboreplay.gym_wrapper import wrap as _wrap
    return _wrap(*args, **kwargs)


__all__ = ["Recorder", "Replay", "compare", "export_csv", "wrap", "__version__"]
