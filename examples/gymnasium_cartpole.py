"""RoboReplay Example: Gymnasium CartPole

Demonstrates using roboreplay.wrap() to automatically record
Gymnasium environment interactions.

Requires: pip install roboreplay[gym]

Run:
    python examples/gymnasium_cartpole.py

Output:
    - Creates cartpole_demo.rrp
    - Prints recording summary
    - Runs diagnosis
"""


def main() -> None:
    try:
        import gymnasium as gym
    except ImportError:
        print("This example requires gymnasium. Install with:")
        print("  pip install roboreplay[gym]")
        return

    import roboreplay
    from roboreplay import Replay
    from roboreplay.diagnose import diagnose

    print("=" * 60)
    print("  RoboReplay Example: Gymnasium CartPole")
    print("=" * 60)
    print()

    # 1. Create environment and wrap it
    print("[1/3] Recording CartPole episodes...")
    env = gym.make("CartPole-v1")
    wrapped = roboreplay.wrap(
        env,
        name="cartpole_demo",
        metadata={"robot": "cartpole", "task": "balance"},
    )

    # Run several episodes
    total_steps = 0
    for episode in range(5):
        obs, info = wrapped.reset()
        episode_steps = 0
        done = False

        while not done:
            action = wrapped.action_space.sample()
            obs, reward, terminated, truncated, info = wrapped.step(action)
            episode_steps += 1
            done = terminated or truncated

        total_steps += episode_steps
        print(f"  Episode {episode + 1}: {episode_steps} steps")

    wrapped.close()
    print(f"  Total steps recorded: {total_steps}")
    print(f"  Saved: {wrapped.recording_path}")
    print()

    # 2. Replay and inspect
    print("[2/3] Replaying recording...")
    replay = Replay(str(wrapped.recording_path))
    print(replay)
    print()

    # Show sample data
    mid = replay.num_steps // 2
    frame = replay[mid]
    print(f"Sample data at step {mid}:")
    for name, value in frame.items():
        print(f"  {name}: {value}")
    print()
    replay.close()

    # 3. Diagnose
    print("[3/3] Running diagnosis...")
    result = diagnose(str(wrapped.recording_path))
    print(f"  Anomalies: {len(result.anomalies)}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Warnings: {len(result.warnings)}")

    if result.has_failures:
        print()
        for a in result.failures[:5]:
            print(f"  [{a.step}] {a.anomaly_type}: {a.description}")

    print()
    print("=" * 60)
    print("  Done! Try:")
    print(f"    roboreplay info {wrapped.recording_path}")
    print(f"    roboreplay diagnose {wrapped.recording_path}")
    print(f"    roboreplay export {wrapped.recording_path} --format html")
    print("=" * 60)


if __name__ == "__main__":
    main()
