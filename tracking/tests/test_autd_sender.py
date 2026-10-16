import unittest
from types import SimpleNamespace

from autd3.commands import Modulation

from core.autd_sender import AutdSender, TargetCommand


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
