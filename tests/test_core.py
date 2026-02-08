"""Tests for RoboReplay core pipeline: record → save → load → replay → diagnose."""

import csv
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from roboreplay import Recorder, Replay
from roboreplay.diagnose import diagnose
from roboreplay.diagnose.anomaly import (
    detect_flatlines,
    detect_sudden_drops,
    detect_sudden_spikes,
)
from roboreplay.export.csv import export_csv
from roboreplay.export.html import _lttb_downsample, export_html
from roboreplay.gym_wrapper import RecordingWrapper, wrap
from roboreplay.storage.reader import Reader
from roboreplay.storage.writer import StreamingWriter
from roboreplay.utils.schema import EventLog, RecordingMetadata, RecordingSchema

# ── Schema Models ──────────────────────────────────────────


class TestSchemaModels:
    def test_recording_metadata_roundtrip(self):
        meta = RecordingMetadata(name="test", robot="panda", task="pick")
        json_str = meta.to_json()
        restored = RecordingMetadata.from_json(json_str)
        assert restored.name == "test"
        assert restored.robot == "panda"

    def test_recording_schema_add_and_validate(self):
        schema = RecordingSchema()
        schema.add_channel("state", "float32", (7,))
        schema.add_channel("reward", "float32", (1,))

        data_ok = np.zeros(7, dtype=np.float32)
        schema.validate_step("state", data_ok)  # Should not raise

        data_bad = np.zeros(3, dtype=np.float32)
        with pytest.raises(ValueError, match="shape mismatch"):
            schema.validate_step("state", data_bad)

    def test_event_log(self):
        log = EventLog()
        log.add(10, "grasp_start", {"force": 5.0})
        log.add(50, "failure", {"type": "slip"})
        log.add(80, "episode_end")

        assert len(log) == 3
        failures = log.where("failure")
        assert len(failures) == 1
        assert failures[0].step == 50

        json_str = log.to_json()
        restored = EventLog.from_json(json_str)
        assert len(restored) == 3

    def test_schema_roundtrip(self):
        schema = RecordingSchema()
        schema.add_channel("state", "float32", (7,))
        json_str = schema.to_json()
        restored = RecordingSchema.from_json(json_str)
        assert "state" in restored.channels
        assert restored.channels["state"].shape == (7,)


# ── Storage Writer/Reader ──────────────────────────────────


class TestStorage:
    def test_write_and_read_basic(self, tmp_path):
        path = tmp_path / "test.rrp"
        meta = RecordingMetadata(name="storage_test")

        # Write
        writer = StreamingWriter(path, meta)
        writer.open()
        for i in range(50):
            writer.write_step({
                "state": np.random.randn(7).astype(np.float32),
                "reward": np.array([float(i) / 50], dtype=np.float32),
            })
        writer.write_event(25, "midpoint", {"note": "halfway"})
        writer.close()

        assert path.exists()

        # Read
        reader = Reader(path)
        reader.open()
        assert reader.num_steps == 50
        assert set(reader.channel_names) == {"state", "reward"}
        assert len(reader.events) == 1
        assert reader.events[0].event_type == "midpoint"

        # Random access
        step_data = reader.get_step(25)
        assert "state" in step_data
        assert step_data["state"].shape == (7,)

        # Slice
        sliced = reader.get_slice(10, 20)
        assert sliced["state"].shape == (10, 7)
        assert sliced["reward"].shape == (10, 1)

        reader.close()

    def test_write_flush_survives(self, tmp_path):
        """Test that periodic flushing writes data to disk."""
        path = tmp_path / "flush_test.rrp"
        meta = RecordingMetadata(name="flush_test")

        writer = StreamingWriter(path, meta)
        writer.open()
        # Write more than one chunk (CHUNK_STEPS=100)
        for i in range(150):
            writer.write_step({"x": np.array([float(i)], dtype=np.float32)})
        writer.close()

        reader = Reader(path)
        reader.open()
        assert reader.num_steps == 150
        reader.close()

    def test_stats_computed(self, tmp_path):
        path = tmp_path / "stats_test.rrp"
        meta = RecordingMetadata(name="stats_test")

        writer = StreamingWriter(path, meta)
        writer.open()
        for i in range(100):
            writer.write_step({"val": np.array([float(i)], dtype=np.float32)})
        writer.close()

        reader = Reader(path)
        reader.open()
        assert "val" in reader.stats
        assert reader.stats["val"].min == 0.0
        assert reader.stats["val"].max == 99.0
        assert reader.stats["val"].num_steps == 100
        reader.close()


# ── Recorder ───────────────────────────────────────────────


class TestRecorder:
    def test_basic_recording(self, tmp_path):
        path = tmp_path / "rec_test.rrp"
        rec = Recorder("rec_test", path=path)
        rec.start()
        for i in range(30):
            rec.step(pos=np.array([1.0, 2.0, 3.0]), vel=np.array([0.1]))
        rec.mark_event("done")
        saved = rec.save()

        assert saved.exists()
        assert rec.num_steps == 30

    def test_context_manager(self, tmp_path):
        path = tmp_path / "ctx_test.rrp"
        with Recorder("ctx_test", path=path) as rec:
            for i in range(10):
                rec.step(x=np.array([float(i)]))

        assert path.exists()
        replay = Replay(path)
        assert replay.num_steps == 10
        replay.close()

    def test_auto_start_on_step(self, tmp_path):
        path = tmp_path / "auto_start.rrp"
        rec = Recorder("auto_start", path=path)
        # Don't call start() — should auto-start on first step
        rec.step(x=np.array([1.0]))
        rec.step(x=np.array([2.0]))
        rec.save()

        replay = Replay(path)
        assert replay.num_steps == 2
        replay.close()

    def test_scalar_coercion(self, tmp_path):
        """Scalars and lists should be auto-converted to numpy arrays."""
        path = tmp_path / "coerce.rrp"
        with Recorder("coerce", path=path) as rec:
            rec.step(reward=0.5, flag=1)  # Python scalars
            rec.step(reward=0.8, flag=0)

        replay = Replay(path)
        assert replay.num_steps == 2
        data = replay[0]
        assert data["reward"].shape == (1,)
        replay.close()

    def test_metadata_preserved(self, tmp_path):
        path = tmp_path / "meta.rrp"
        with Recorder("meta", path=path, metadata={"robot": "panda", "task": "pick"}) as rec:
            rec.step(x=np.array([1.0]))

        replay = Replay(path)
        assert replay.robot == "panda"
        assert replay.task == "pick"
        replay.close()

    def test_empty_step_raises(self, tmp_path):
        path = tmp_path / "empty.rrp"
        rec = Recorder("empty", path=path)
        rec.start()
        with pytest.raises(ValueError, match="at least one channel"):
            rec.step()

    def test_events_preserved(self, tmp_path):
        path = tmp_path / "events.rrp"
        with Recorder("events", path=path) as rec:
            for i in range(20):
                rec.step(x=np.array([float(i)]))
                if i == 10:
                    rec.mark_event("halfway", {"step": 10})

        replay = Replay(path)
        assert len(replay.events) == 1
        assert replay.events[0].event_type == "halfway"
        assert replay.events[0].data["step"] == 10
        replay.close()


# ── Replay ─────────────────────────────────────────────────


class TestReplay:
    @pytest.fixture
    def sample_recording(self, tmp_path):
        """Create a sample recording for replay tests."""
        path = tmp_path / "sample.rrp"
        with Recorder("sample", path=path, metadata={"robot": "test_bot"}) as rec:
            for i in range(100):
                rec.step(
                    pos=np.array([float(i), float(i) * 2]),
                    vel=np.array([0.1 * i]),
                    reward=float(i) / 100,
                )
                if i == 50:
                    rec.mark_event("midpoint")
        return path

    def test_basic_replay(self, sample_recording):
        r = Replay(sample_recording)
        assert r.num_steps == 100
        assert r.name == "sample"
        assert r.robot == "test_bot"
        assert set(r.channels) == {"pos", "vel", "reward"}
        r.close()

    def test_indexing(self, sample_recording):
        r = Replay(sample_recording)
        frame = r[50]
        assert "pos" in frame
        assert frame["pos"].shape == (2,)
        np.testing.assert_allclose(frame["pos"], [50.0, 100.0], atol=0.01)
        r.close()

    def test_negative_indexing(self, sample_recording):
        r = Replay(sample_recording)
        last = r[-1]
        assert "pos" in last
        np.testing.assert_allclose(last["pos"], [99.0, 198.0], atol=0.01)
        r.close()

    def test_slicing(self, sample_recording):
        r = Replay(sample_recording)
        chunk = r[10:20]
        assert chunk["pos"].shape == (10, 2)
        assert chunk["vel"].shape == (10, 1)
        r.close()

    def test_channel_access(self, sample_recording):
        r = Replay(sample_recording)
        all_pos = r.channel("pos")
        assert all_pos.shape == (100, 2)

        partial = r.channel("pos", start=40, end=60)
        assert partial.shape == (20, 2)
        r.close()

    def test_events_accessible(self, sample_recording):
        r = Replay(sample_recording)
        assert len(r.events) == 1
        assert r.events[0].event_type == "midpoint"
        r.close()

    def test_len(self, sample_recording):
        r = Replay(sample_recording)
        assert len(r) == 100
        r.close()

    def test_summary_string(self, sample_recording):
        r = Replay(sample_recording)
        s = r.summary()
        assert "sample" in s
        assert "test_bot" in s
        assert "100" in s
        r.close()

    def test_repr(self, sample_recording):
        r = Replay(sample_recording)
        rep = repr(r)
        assert "sample" in rep
        r.close()

    def test_context_manager(self, sample_recording):
        with Replay(sample_recording) as r:
            assert r.num_steps == 100


# ── Anomaly Detection ──────────────────────────────────────


class TestAnomalyDetection:
    def test_detect_sudden_drop(self):
        # Signal stable at 10.0, then drops to 2.0 at step 50
        data = np.ones(100) * 10.0
        data[50:] = 2.0

        anomalies = detect_sudden_drops(data, "force", threshold_pct=0.5, window=5)
        assert len(anomalies) >= 1
        assert any(a.step == 50 for a in anomalies)

    def test_no_drop_in_stable_signal(self):
        data = np.ones(100) * 5.0 + np.random.normal(0, 0.01, 100)
        anomalies = detect_sudden_drops(data, "force", threshold_pct=0.5)
        assert len(anomalies) == 0

    def test_detect_spike(self):
        data = np.random.normal(0, 1, 200)
        data[100] = 50.0  # Huge spike

        anomalies = detect_sudden_spikes(data, "sensor", threshold_std=5.0)
        assert len(anomalies) >= 1
        assert any(a.step == 100 for a in anomalies)

    def test_detect_flatline(self):
        data = np.random.normal(5, 0.5, 200)
        data[100:170] = 5.0  # Flatline for 70 steps

        anomalies = detect_flatlines(data, "sensor", min_duration=50)
        assert len(anomalies) >= 1
        assert any(a.details.get("duration", 0) >= 50 for a in anomalies)

    def test_no_flatline_for_always_constant(self):
        """A signal that's always constant shouldn't be flagged."""
        data = np.ones(200) * 3.14
        anomalies = detect_flatlines(data, "constant_signal", min_duration=50)
        assert len(anomalies) == 0


# ── Diagnosis ──────────────────────────────────────────────


class TestDiagnosis:
    def test_clean_recording(self, tmp_path):
        """A smooth recording should produce no failures."""
        path = tmp_path / "clean.rrp"
        rng = np.random.default_rng(42)
        with Recorder("clean", path=path) as rec:
            for i in range(200):
                rec.step(
                    pos=np.array([5.0 + rng.normal(0, 0.05)]),
                    vel=np.array([1.0 + rng.normal(0, 0.02)]),
                )

        result = diagnose(path)
        assert not result.has_failures

    def test_recording_with_failure(self, tmp_path):
        """A recording with a sudden force drop should detect it."""
        path = tmp_path / "failure.rrp"
        with Recorder("failure", path=path) as rec:
            for i in range(200):
                if i < 100:
                    force = 8.0 + np.random.normal(0, 0.1)
                elif i == 100:
                    force = 0.5  # Sudden drop
                else:
                    force = 0.5 + np.random.normal(0, 0.05)
                rec.step(force=np.array([force]))

        result = diagnose(path)
        assert result.has_failures
        # The force drop should be among the top anomalies
        force_anomalies = [a for a in result.failures if a.channel == "force"]
        assert len(force_anomalies) >= 1

    def test_diagnosis_result_properties(self, tmp_path):
        path = tmp_path / "props.rrp"
        with Recorder("props", path=path) as rec:
            for i in range(50):
                rec.step(x=np.array([1.0]))

        result = diagnose(path)
        assert result.recording_name == "props"
        assert result.num_steps == 50


# ── Compare ─────────────────────────────────────────────────


class TestCompare:
    def test_compare_identical(self, tmp_path):
        """Comparing a recording to itself should show no divergence."""
        path = tmp_path / "same.rrp"
        with Recorder("same", path=path) as rec:
            for i in range(100):
                rec.step(x=np.array([float(i) * 0.1]))

        from roboreplay import compare

        result = compare(path, path)
        assert result.divergence_step is None
        assert "x" in result.channel_diffs

    def test_compare_different(self, tmp_path):
        """Two different recordings should show divergence."""
        path_a = tmp_path / "a.rrp"
        path_b = tmp_path / "b.rrp"

        rng = np.random.default_rng(42)
        with Recorder("run_a", path=path_a) as rec:
            for i in range(100):
                rec.step(force=np.array([8.0 + rng.normal(0, 0.1)]))

        with Recorder("run_b", path=path_b) as rec:
            for i in range(100):
                if i < 50:
                    rec.step(force=np.array([8.0 + rng.normal(0, 0.1)]))
                else:
                    rec.step(force=np.array([2.0 + rng.normal(0, 0.1)]))

        from roboreplay import compare

        result = compare(path_a, path_b)
        assert result.divergence_step is not None
        assert result.divergence_step >= 45 and result.divergence_step <= 55
        summary = result.summary()
        assert "run_a" in summary
        assert "run_b" in summary


# ── CLI ────────────────────────────────────────────────────


class TestCLI:
    def test_cli_info(self, tmp_path):
        from click.testing import CliRunner

        from roboreplay.cli.main import cli

        path = tmp_path / "cli_test.rrp"
        with Recorder("cli_test", path=path) as rec:
            for i in range(20):
                rec.step(x=np.array([float(i)]))

        runner = CliRunner()
        result = runner.invoke(cli, ["info", str(path)])
        assert result.exit_code == 0
        assert "cli_test" in result.output
        assert "20" in result.output

    def test_cli_diagnose(self, tmp_path):
        from click.testing import CliRunner

        from roboreplay.cli.main import cli

        path = tmp_path / "cli_diag.rrp"
        with Recorder("cli_diag", path=path) as rec:
            for i in range(50):
                rec.step(x=np.array([1.0]))

        runner = CliRunner()
        result = runner.invoke(cli, ["diagnose", str(path)])
        assert result.exit_code == 0

    def test_cli_version(self):
        from click.testing import CliRunner

        from roboreplay.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert "0.1.0" in result.output

    def test_cli_export_csv(self, tmp_path):
        from click.testing import CliRunner

        from roboreplay.cli.main import cli

        path = tmp_path / "cli_export.rrp"
        with Recorder("cli_export", path=path) as rec:
            for i in range(20):
                rec.step(x=np.array([float(i)]))

        runner = CliRunner()
        out_dir = tmp_path / "csv_out"
        out_dir.mkdir()
        result = runner.invoke(cli, ["export", str(path), "--format", "csv", "-o", str(out_dir)])
        assert result.exit_code == 0
        assert "CSV" in result.output

    def test_cli_export_html(self, tmp_path):
        from click.testing import CliRunner

        from roboreplay.cli.main import cli

        path = tmp_path / "cli_html.rrp"
        with Recorder("cli_html", path=path) as rec:
            for i in range(20):
                rec.step(x=np.array([float(i)]))

        runner = CliRunner()
        out_file = tmp_path / "output.html"
        result = runner.invoke(cli, ["export", str(path), "--format", "html", "-o", str(out_file)])
        assert result.exit_code == 0
        assert "HTML" in result.output
        assert out_file.exists()

    def test_cli_compare(self, tmp_path):
        from click.testing import CliRunner

        from roboreplay.cli.main import cli

        path_a = tmp_path / "cmp_a.rrp"
        path_b = tmp_path / "cmp_b.rrp"
        with Recorder("cmp_a", path=path_a) as rec:
            for i in range(50):
                rec.step(x=np.array([float(i)]))
        with Recorder("cmp_b", path=path_b) as rec:
            for i in range(50):
                rec.step(x=np.array([float(i) * 2]))

        runner = CliRunner()
        result = runner.invoke(cli, ["compare", str(path_a), str(path_b)])
        assert result.exit_code == 0
        assert "cmp_a" in result.output


# ── CSV Export ───────────────────────────────────────────


class TestCSVExport:
    @pytest.fixture
    def sample_recording(self, tmp_path):
        path = tmp_path / "csv_sample.rrp"
        with Recorder("csv_sample", path=path, metadata={"robot": "test"}) as rec:
            for i in range(50):
                rec.step(
                    pos=np.array([float(i), float(i) * 2, float(i) * 3]),
                    vel=np.array([0.1 * i]),
                    reward=float(i) / 50,
                )
                if i == 25:
                    rec.mark_event("midpoint", {"note": "halfway"})
        return path

    def test_export_creates_files(self, sample_recording, tmp_path):
        out_dir = tmp_path / "csv_out"
        created = export_csv(sample_recording, output_dir=out_dir)
        assert len(created) >= 2  # channels + metadata at minimum
        for f in created:
            assert f.exists()

    def test_channels_csv_content(self, sample_recording, tmp_path):
        out_dir = tmp_path / "csv_out"
        export_csv(sample_recording, output_dir=out_dir)
        channels_file = out_dir / "csv_sample_channels.csv"
        assert channels_file.exists()

        with open(channels_file) as f:
            reader = csv.reader(f)
            headers = next(reader)
            assert headers[0] == "step"
            assert "pos_0" in headers
            assert "pos_1" in headers
            assert "pos_2" in headers
            assert "vel" in headers or "vel_0" in headers
            rows = list(reader)
            assert len(rows) == 50

    def test_events_csv_content(self, sample_recording, tmp_path):
        out_dir = tmp_path / "csv_out"
        export_csv(sample_recording, output_dir=out_dir)
        events_file = out_dir / "csv_sample_events.csv"
        assert events_file.exists()

        with open(events_file) as f:
            reader = csv.reader(f)
            headers = next(reader)
            assert "step" in headers
            assert "event_type" in headers
            rows = list(reader)
            assert len(rows) == 1
            assert rows[0][1] == "midpoint"

    def test_metadata_csv_content(self, sample_recording, tmp_path):
        out_dir = tmp_path / "csv_out"
        export_csv(sample_recording, output_dir=out_dir)
        meta_file = out_dir / "csv_sample_metadata.csv"
        assert meta_file.exists()

        with open(meta_file) as f:
            reader = csv.reader(f)
            headers = next(reader)
            assert headers == ["key", "value"]
            rows = list(reader)
            keys = [r[0] for r in rows]
            assert "name" in keys
            assert "robot" in keys

    def test_export_specific_channels(self, sample_recording, tmp_path):
        out_dir = tmp_path / "csv_specific"
        export_csv(sample_recording, output_dir=out_dir, channels=["pos"])
        channels_file = out_dir / "csv_sample_channels.csv"

        with open(channels_file) as f:
            reader = csv.reader(f)
            headers = next(reader)
            assert "pos_0" in headers
            # vel and reward should not be present
            assert all("vel" not in h and "reward" not in h for h in headers)

    def test_export_no_events(self, tmp_path):
        path = tmp_path / "no_events.rrp"
        with Recorder("no_events", path=path) as rec:
            for i in range(10):
                rec.step(x=np.array([float(i)]))

        out_dir = tmp_path / "csv_no_events"
        created = export_csv(path, output_dir=out_dir)
        # Should not create events file
        filenames = [f.name for f in created]
        assert "no_events_events.csv" not in filenames


# ── HTML Export ──────────────────────────────────────────


class TestHTMLExport:
    @pytest.fixture
    def sample_recording(self, tmp_path):
        path = tmp_path / "html_sample.rrp"
        with Recorder("html_sample", path=path, metadata={"robot": "test"}) as rec:
            for i in range(100):
                rec.step(
                    pos=np.array([float(i), float(i) * 2]),
                    reward=float(i) / 100,
                )
                if i == 50:
                    rec.mark_event("midpoint")
        return path

    def test_export_creates_html(self, sample_recording, tmp_path):
        out_path = tmp_path / "output.html"
        result = export_html(sample_recording, output=out_path)
        assert result.exists()
        assert result.suffix == ".html"

    def test_html_contains_chart_data(self, sample_recording, tmp_path):
        out_path = tmp_path / "output.html"
        export_html(sample_recording, output=out_path)
        content = out_path.read_text(encoding="utf-8")
        assert "Chart.js" in content or "chart.js" in content
        assert "html_sample" in content
        assert "pos" in content
        assert "reward" in content

    def test_html_contains_events(self, sample_recording, tmp_path):
        out_path = tmp_path / "output.html"
        export_html(sample_recording, output=out_path)
        content = out_path.read_text(encoding="utf-8")
        assert "midpoint" in content

    def test_html_specific_channels(self, sample_recording, tmp_path):
        out_path = tmp_path / "output.html"
        export_html(sample_recording, output=out_path, channels=["reward"])
        content = out_path.read_text(encoding="utf-8")
        assert "reward" in content

    def test_lttb_downsample_passthrough(self):
        """Small data should pass through without downsampling."""
        data = np.arange(10, dtype=np.float64)
        indices, values = _lttb_downsample(data, threshold=20)
        assert len(indices) == 10
        assert len(values) == 10

    def test_lttb_downsample_reduces(self):
        """Large data should be downsampled."""
        data = np.sin(np.linspace(0, 10, 5000))
        indices, values = _lttb_downsample(data, threshold=500)
        assert len(indices) == 500
        assert len(values) == 500
        # First and last points preserved
        assert indices[0] == 0
        assert indices[-1] == 4999

    def test_html_default_output_path(self, sample_recording):
        result = export_html(sample_recording)
        assert result.exists()
        assert result.suffix == ".html"
        result.unlink()  # Clean up


# ── Gymnasium Wrapper ────────────────────────────────────


class MockEnv:
    """Minimal mock of a Gymnasium environment."""

    def __init__(self):
        self.action_space = MagicMock()
        self.action_space.sample.return_value = np.array([1.0])
        self.observation_space = MagicMock()
        self._step_count = 0
        self._closed = False

    def reset(self, **kwargs):
        self._step_count = 0
        obs = np.array([0.1, 0.2, 0.3, 0.4])
        return obs, {"info_key": "value"}

    def step(self, action):
        self._step_count += 1
        obs = np.array([0.1 * self._step_count, 0.2, 0.3, 0.4])
        reward = 1.0
        terminated = self._step_count >= 10
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def close(self):
        self._closed = True


class TestGymWrapper:
    def test_wrap_records_steps(self, tmp_path):
        env = MockEnv()
        path = tmp_path / "gym_test.rrp"
        wrapped = wrap(env, name="gym_test", path=path)

        obs, info = wrapped.reset()
        for _ in range(5):
            action = np.array([1.0])
            obs, reward, terminated, truncated, info = wrapped.step(action)
        wrapped.close()

        assert path.exists()
        replay = Replay(path)
        assert replay.num_steps == 5
        assert "observation" in replay.channels
        assert "action" in replay.channels
        assert "reward" in replay.channels
        replay.close()

    def test_wrap_records_events(self, tmp_path):
        env = MockEnv()
        path = tmp_path / "gym_events.rrp"
        wrapped = wrap(env, name="gym_events", path=path)

        # First episode
        obs, info = wrapped.reset()
        for _ in range(10):
            obs, reward, terminated, truncated, info = wrapped.step(np.array([1.0]))
            if terminated:
                break

        # Second episode
        obs, info = wrapped.reset()
        for _ in range(5):
            obs, reward, terminated, truncated, info = wrapped.step(np.array([1.0]))

        wrapped.close()

        replay = Replay(path)
        event_types = [e.event_type for e in replay.events.events]
        assert "episode_terminated" in event_types
        assert "episode_reset" in event_types
        assert "recording_end" in event_types
        replay.close()

    def test_wrap_delegates_attributes(self, tmp_path):
        env = MockEnv()
        path = tmp_path / "gym_delegate.rrp"
        wrapped = wrap(env, name="gym_delegate", path=path)
        assert wrapped.action_space is env.action_space
        assert wrapped.observation_space is env.observation_space
        wrapped.close()

    def test_wrap_context_manager(self, tmp_path):
        env = MockEnv()
        path = tmp_path / "gym_ctx.rrp"
        with RecordingWrapper(env, name="gym_ctx", path=path) as wrapped:
            obs, info = wrapped.reset()
            wrapped.step(np.array([1.0]))
            wrapped.step(np.array([1.0]))

        assert path.exists()
        replay = Replay(path)
        assert replay.num_steps == 2
        replay.close()

    def test_wrap_metadata(self, tmp_path):
        env = MockEnv()
        path = tmp_path / "gym_meta.rrp"
        wrapped = wrap(env, name="gym_meta", path=path, metadata={"robot": "cartpole"})
        obs, info = wrapped.reset()
        wrapped.step(np.array([1.0]))
        wrapped.close()

        replay = Replay(path)
        assert replay.metadata.task == "gymnasium"
        replay.close()

    def test_wrap_recording_path(self, tmp_path):
        env = MockEnv()
        path = tmp_path / "gym_path.rrp"
        wrapped = wrap(env, name="gym_path", path=path)
        assert wrapped.recording_path == path
        wrapped.close()


# ── LLM Diagnosis ────────────────────────────────────────


class TestLLMDiagnosis:
    def test_build_prompt(self):
        from roboreplay.diagnose.anomaly import Anomaly
        from roboreplay.diagnose.llm import _build_prompt
        from roboreplay.utils.schema import ChannelStats, EventLog, RecordingMetadata

        meta = RecordingMetadata(name="test", robot="panda", task="pick")
        anomalies = [
            Anomaly(
                channel="force",
                step=100,
                anomaly_type="sudden_drop",
                severity=0.8,
                description="force dropped 80% at step 100",
            )
        ]
        stats = {
            "force": ChannelStats(name="force", min=0.0, max=10.0, mean=5.0, std=2.0, num_steps=200)
        }
        events = EventLog()
        events.add(50, "grasp_start")

        prompt = _build_prompt(meta, anomalies, stats, events)
        assert "panda" in prompt
        assert "pick" in prompt
        assert "sudden_drop" in prompt
        assert "grasp_start" in prompt

    def test_parse_response(self):
        from roboreplay.diagnose.llm import _parse_response

        response = """
EXPLANATION: The robot dropped the object during lifting due to insufficient grip force.

ROOT CAUSES:
1. Grip force was below the threshold needed to maintain grasp during acceleration.
2. The lifting acceleration increased too rapidly.

RECOMMENDATIONS:
1. Increase the grip force setpoint during lifting phases.
2. Implement adaptive force control that responds to lifting acceleration.
3. Add a force threshold alarm before the slip occurs.
"""
        result = _parse_response(response)
        assert "dropped the object" in result.explanation
        assert len(result.root_causes) == 2
        assert len(result.recommendations) == 3
        assert "grip force" in result.root_causes[0].lower() or "Grip" in result.root_causes[0]

    def test_parse_empty_response(self):
        from roboreplay.diagnose.llm import _parse_response

        result = _parse_response("")
        assert result.explanation == ""
        assert result.root_causes == []
        assert result.recommendations == []

    def test_llm_diagnose_import_error(self):
        """llm_diagnose should raise ImportError when anthropic is not installed."""
        from roboreplay.diagnose.llm import llm_diagnose
        from roboreplay.utils.schema import EventLog, RecordingMetadata

        meta = RecordingMetadata(name="test")
        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(ImportError, match="anthropic"):
                llm_diagnose(meta, [], {}, EventLog())

    def test_diagnosis_result_llm_field(self, tmp_path):
        """DiagnosisResult should have llm_result field."""
        path = tmp_path / "llm_field.rrp"
        with Recorder("llm_field", path=path) as rec:
            for i in range(50):
                rec.step(x=np.array([1.0]))

        result = diagnose(path)
        assert result.llm_result is None  # Not enabled
