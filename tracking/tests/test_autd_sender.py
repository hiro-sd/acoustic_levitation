import unittest
import threading
import csv
import tempfile
from collections import deque
from pathlib import Path
from types import SimpleNamespace

from autd3.commands import Modulation

from core.autd_sender import (
    AutdSender,
    AUTD_UPDATE_LOG_HEADER,
    SentCommandTelemetry,
    TargetCommand,
    quantize_intensity_ratio,
)


class FakeAutd:
    def __init__(self, error=None):
        self.error = error
        self.sent = []

    def send(self, datagram):
        self.sent.append(datagram)
        if self.error is not None:
            raise self.error


def make_sender(*, last_intensity=None, autd=None):
    sender = AutdSender.__new__(AutdSender)
    sender.autd = FakeAutd() if autd is None else autd
    sender.cfg = SimpleNamespace(intensity_update_eps=0.005)
    sender._last_intensity_ratio = last_intensity
    sender._lock = threading.Lock()
    sender.command_latency_ema_ms = None
    sender.initial_capture_latency_ema_ms = None
    return sender


def make_target(intensity_ratio):
    return TargetCommand(
        x=0.0,
        y=0.0,
        z=400.0,
        enqueued_time_sec=0.0,
        intensity_ratio=intensity_ratio,
    )


class AutdSenderIntensityIntegrationTests(unittest.TestCase):
    def test_sent_update_log_contains_timing_and_quantized_intensity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "updates.csv"
            sender = make_sender()
            sender._update_session_id = "session"
            sender._update_log_file = path.open("w", newline="", encoding="utf-8")
            sender._update_log_writer = csv.writer(sender._update_log_file)
            sender._update_log_writer.writerow(AUTD_UPDATE_LOG_HEADER)
            sender._update_log_rows_since_flush = 0
            sender._update_log_last_flush_sec = 0.0
            target = TargetCommand(
                x=1.0,
                y=2.0,
                z=3.0,
                enqueued_time_sec=10.000,
                intensity_ratio=0.6,
                measurement_time_sec=9.990,
                measurement_age_sec=0.010,
                estimated_actuation_delay_sec=0.003,
                prediction_horizon_sec=0.013,
                control_mode="CAPTURE_BRAKING",
                release_timestamp_ms=9990,
            )
            telemetry = SentCommandTelemetry(
                sequence=7,
                dequeued_time_sec=10.001,
                send_started_time_sec=10.002,
                sent_time_sec=10.004,
                target=target,
                modulation_byte=153,
                quantized_intensity_ratio=0.6,
            )
            sender._write_update_telemetry(telemetry)
            sender._update_log_file.close()

            with path.open(newline="", encoding="utf-8") as file:
                row = next(csv.DictReader(file))
            self.assertEqual(row["command_sequence"], "7")
            self.assertEqual(row["control_mode"], "CAPTURE_BRAKING")
            self.assertEqual(row["modulation_byte"], "153")
            self.assertAlmostEqual(float(row["measurement_age_ms"]), 10.0)
            self.assertAlmostEqual(float(row["prediction_horizon_ms"]), 13.0)
            self.assertAlmostEqual(
                float(row["prediction_horizon_shortfall_ms"]),
                1.0,
            )

    def test_intensity_quantization_matches_8_bit_modulation(self):
        modulation_byte, quantized = quantize_intensity_ratio(0.6)
        self.assertEqual(modulation_byte, 153)
        self.assertAlmostEqual(quantized, 0.6)

    def test_exact_sent_lookup_does_not_accept_a_later_sequence(self):
        sender = make_sender()
        later = SentCommandTelemetry(
            sequence=11,
            dequeued_time_sec=1.0,
            send_started_time_sec=1.1,
            sent_time_sec=1.2,
            target=make_target(0.7),
        )
        sender._sent_history = deque([later], maxlen=256)
        self.assertIsNone(sender.get_sent_telemetry(10))
        self.assertIs(sender.get_sent_telemetry(11), later)

    def test_command_latency_uses_fallback_then_runtime_ema(self):
        sender = make_sender()

        self.assertAlmostEqual(sender.estimated_command_latency_sec(0.0075), 0.0075)
        sender.command_latency_ema_ms = 3.25
        self.assertAlmostEqual(sender.estimated_command_latency_sec(0.0075), 0.00325)

    def test_initial_capture_latency_has_independent_fallback_and_ema(self):
        sender = make_sender()

        self.assertAlmostEqual(
            sender.estimated_initial_capture_latency_sec(0.0035),
            0.0035,
        )
        self.assertAlmostEqual(sender.initial_capture_latency_ema_ms, 3.5)
        sender.command_latency_ema_ms = 2.25
        self.assertAlmostEqual(
            sender.estimated_initial_capture_latency_sec(0.0035),
            0.0035,
        )
        sender.initial_capture_latency_ema_ms = 3.75
        self.assertAlmostEqual(
            sender.estimated_initial_capture_latency_sec(0.0035),
            0.00375,
        )

    def test_only_initial_commands_update_initial_latency_ema(self):
        sender = make_sender()
        normal = make_target(0.6)
        initial = TargetCommand(
            x=0.0,
            y=0.0,
            z=400.0,
            enqueued_time_sec=0.0,
            intensity_ratio=0.8,
            is_initial_capture_command=True,
        )

        sender.estimated_initial_capture_latency_sec(0.0035)
        sender._record_command_latency_ms(normal, 2.0)
        self.assertAlmostEqual(sender.command_latency_ema_ms, 2.0)
        self.assertAlmostEqual(sender.initial_capture_latency_ema_ms, 3.5)

        sender._record_command_latency_ms(initial, 4.0)
        self.assertAlmostEqual(sender.command_latency_ema_ms, 2.2)
        self.assertAlmostEqual(sender.initial_capture_latency_ema_ms, 3.55)

        sender._record_command_latency_ms(initial, 3.0)
        self.assertAlmostEqual(sender.command_latency_ema_ms, 2.28)
        self.assertAlmostEqual(sender.initial_capture_latency_ema_ms, 3.495)

    def test_existing_target_defaults_to_unweighted_stm(self):
        self.assertIsNone(make_target(0.6).focus_dwell_counts)

    def test_changed_intensity_is_combined_with_stm(self):
        sender = make_sender(last_intensity=0.6)
        stm = object()

        datagram, pending_ratio = sender._combine_stm_with_intensity_if_needed(
            stm,
            make_target(0.8),
        )

        self.assertIsInstance(datagram, tuple)
        self.assertEqual(len(datagram), 2)
        self.assertIsInstance(datagram[0], Modulation)
        self.assertIs(datagram[1], stm)
        self.assertEqual(pending_ratio, 0.8)
        self.assertEqual(sender._last_intensity_ratio, 0.6)

        sender._send_field_datagram(datagram, pending_ratio)

        self.assertEqual(len(sender.autd.sent), 1)
        self.assertIs(sender.autd.sent[0], datagram)
        self.assertEqual(sender._last_intensity_ratio, 0.8)

    def test_unchanged_intensity_sends_only_stm(self):
        sender = make_sender(last_intensity=0.6)
        stm = object()

        datagram, pending_ratio = sender._combine_stm_with_intensity_if_needed(
            stm,
            make_target(0.602),
        )
        sender._send_field_datagram(datagram, pending_ratio)

        self.assertIs(datagram, stm)
        self.assertIsNone(pending_ratio)
        self.assertEqual(sender.autd.sent, [stm])
        self.assertEqual(sender._last_intensity_ratio, 0.6)

    def test_failed_send_does_not_mark_intensity_as_sent(self):
        autd = FakeAutd(error=RuntimeError("send failed"))
        sender = make_sender(last_intensity=0.6, autd=autd)
        stm = object()
        datagram, pending_ratio = sender._combine_stm_with_intensity_if_needed(
            stm,
            make_target(0.8),
        )

        with self.assertRaisesRegex(RuntimeError, "send failed"):
            sender._send_field_datagram(datagram, pending_ratio)

        self.assertEqual(sender._last_intensity_ratio, 0.6)


if __name__ == "__main__":
    unittest.main()
