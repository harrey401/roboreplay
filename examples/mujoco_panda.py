"""RoboReplay Example: MuJoCo Panda Arm Reaching Task

Records a Franka Emika Panda robot performing a reaching task using
a simple PD controller. Demonstrates recording real MuJoCo simulation
data with RoboReplay.

Requires: pip install roboreplay[mujoco]

Run:
    python examples/mujoco_panda.py

Output:
    - Creates mujoco_panda_reach.rrp
    - Prints recording summary
    - Runs diagnosis
"""

from __future__ import annotations

import sys

import numpy as np

try:
    import mujoco
except ImportError:
    print("This example requires MuJoCo:")
    print("  pip install roboreplay[mujoco]")
    print()
    print("MuJoCo is free and open-source since 2022.")
    print("See: https://mujoco.readthedocs.io/en/stable/python.html")
    sys.exit(1)

from roboreplay import Recorder, Replay
from roboreplay.diagnose import diagnose

# Panda arm XML — minimal model for reaching task
PANDA_XML = """
<mujoco model="panda_reach">
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <default>
    <joint damping="1" armature="0.1"/>
    <geom condim="1" friction="1 0.005 0.0001"/>
  </default>

  <worldbody>
    <!-- Ground plane -->
    <geom type="plane" size="1 1 0.01" rgba="0.9 0.9 0.9 1"/>

    <!-- Target marker -->
    <body name="target" pos="0.5 0.2 0.4">
      <geom type="sphere" size="0.02" rgba="1 0 0 0.5" contype="0" conaffinity="0"/>
    </body>

    <!-- Panda arm (simplified 7-DOF) -->
    <body name="link0" pos="0 0 0">
      <geom type="cylinder" size="0.06 0.05" rgba="0.9 0.9 0.9 1" mass="4"/>

      <body name="link1" pos="0 0 0.1">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.04" rgba="0.9 0.9 0.9 1" mass="3"/>

        <body name="link2" pos="0 0 0.2">
          <joint name="joint2" type="hinge" axis="0 1 0" range="-1.7628 1.7628"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.04" rgba="0.9 0.9 0.9 1" mass="3"/>

          <body name="link3" pos="0 0 0.2">
            <joint name="joint3" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.15" size="0.035" rgba="0.9 0.9 0.9 1" mass="2"/>

            <body name="link4" pos="0 0 0.15">
              <joint name="joint4" type="hinge" axis="0 1 0" range="-3.0718 -0.0698"/>
              <geom type="capsule" fromto="0 0 0 0 0 0.15" size="0.035" rgba="0.9 0.9 0.9 1" mass="2"/>

              <body name="link5" pos="0 0 0.15">
                <joint name="joint5" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
                <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.03" rgba="0.9 0.9 0.9 1" mass="1.5"/>

                <body name="link6" pos="0 0 0.1">
                  <joint name="joint6" type="hinge" axis="0 1 0" range="-0.0175 3.7525"/>
                  <geom type="capsule" fromto="0 0 0 0 0 0.08" size="0.025" rgba="0.9 0.9 0.9 1" mass="1"/>

                  <body name="link7" pos="0 0 0.08">
                    <joint name="joint7" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
                    <geom type="capsule" fromto="0 0 0 0 0 0.05" size="0.02" rgba="0.3 0.3 0.3 1" mass="0.5"/>

                    <!-- End-effector site for position tracking -->
                    <site name="ee" pos="0 0 0.05" size="0.01"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor joint="joint1" ctrlrange="-87 87" gear="1"/>
    <motor joint="joint2" ctrlrange="-87 87" gear="1"/>
    <motor joint="joint3" ctrlrange="-87 87" gear="1"/>
    <motor joint="joint4" ctrlrange="-87 87" gear="1"/>
    <motor joint="joint5" ctrlrange="-12 12" gear="1"/>
    <motor joint="joint6" ctrlrange="-12 12" gear="1"/>
    <motor joint="joint7" ctrlrange="-12 12" gear="1"/>
  </actuator>
</mujoco>
"""

# Target joint configuration for the reaching task
TARGET_QPOS = np.array([0.4, 0.3, -0.2, -1.8, 0.1, 1.5, 0.3])

# PD controller gains
KP = np.array([600, 600, 600, 600, 250, 150, 50], dtype=np.float64)
KD = np.array([50, 50, 50, 50, 20, 15, 5], dtype=np.float64)


def get_ee_position(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Get end-effector position from the 'ee' site."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
    return data.site_xpos[site_id].copy()


def run_reaching_task(num_steps: int = 2000) -> str:
    """Run a reaching task and record with RoboReplay.

    Returns the path to the saved recording.
    """
    # Load model
    model = mujoco.MjModel.from_xml_string(PANDA_XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Get target position for event logging
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    target_pos = data.xpos[target_body_id].copy()

    rec = Recorder(
        "mujoco_panda_reach",
        metadata={
            "robot": "franka_panda",
            "task": "reaching",
            "controller": "PD",
            "timestep": model.opt.timestep,
            "target_position": target_pos.tolist(),
        },
    )
    rec.start()
    rec.mark_event("phase_start", {"phase": "reaching", "target": target_pos.tolist()})

    reached = False

    for i in range(num_steps):
        # PD controller: torque = Kp * (q_target - q) - Kd * qdot
        q_error = TARGET_QPOS - data.qpos[:7]
        ctrl = KP * q_error - KD * data.qvel[:7]
        data.ctrl[:7] = np.clip(ctrl, -model.actuator_ctrlrange[:, 1], model.actuator_ctrlrange[:, 1])

        # Step simulation (10 substeps per recorded step for stability)
        mujoco.mj_step(model, data)

        # Record every 10th physics step (= 50 Hz recording at 0.002s timestep)
        if i % 10 != 0:
            continue

        ee_pos = get_ee_position(model, data)
        ee_error = np.linalg.norm(ee_pos - target_pos)

        rec.step(
            qpos=data.qpos[:7].copy(),
            qvel=data.qvel[:7].copy(),
            ctrl=data.ctrl[:7].copy(),
            ee_position=ee_pos,
            ee_error=np.array([ee_error]),
        )

        # Check if target reached
        if not reached and ee_error < 0.05:
            reached = True
            rec.mark_event("target_reached", {
                "step": i,
                "ee_position": ee_pos.tolist(),
                "ee_error": float(ee_error),
            })

    rec.mark_event("episode_end", {"success": reached, "final_error": float(ee_error)})
    path = rec.save()
    return str(path)


def main() -> None:
    print("=" * 60)
    print("  RoboReplay Demo: MuJoCo Panda Reaching Task")
    print("=" * 60)
    print()

    # 1. Record
    print("[1/3] Running MuJoCo simulation + recording...")
    path = run_reaching_task()
    print(f"  Saved: {path}")
    print()

    # 2. Replay and inspect
    print("[2/3] Replaying recording...")
    replay = Replay(path)
    print(replay)
    print()

    # Show end-effector trajectory summary
    ee_error = replay.channel("ee_error")
    print(f"End-effector error: start={ee_error[0, 0]:.4f}, end={ee_error[-1, 0]:.4f}")
    print()

    # Show events
    for event in replay.events:
        print(f"  Event: {event.event_type} — {event.data}")
    print()

    replay.close()

    # 3. Diagnose
    print("[3/3] Running diagnosis...")
    result = diagnose(path)
    print()
    if result.has_failures:
        print(f"Found {len(result.failures)} failure(s):")
        for a in result.failures:
            print(f"  [{a.step}] {a.anomaly_type}: {a.description}")
    elif result.warnings:
        print(f"Found {len(result.warnings)} warning(s):")
        for a in result.warnings:
            print(f"  [{a.step}] {a.anomaly_type}: {a.description}")
    else:
        print("No anomalies detected — clean run!")

    print()
    print("=" * 60)
    print("  Demo complete! Try the CLI:")
    print(f"    roboreplay info {path}")
    print(f"    roboreplay diagnose {path}")
    print(f"    roboreplay export {path} --format html")
    print("=" * 60)


if __name__ == "__main__":
    main()
