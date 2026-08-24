import unittest
import sys
from types import ModuleType
from types import SimpleNamespace


class FakeStatic:
    def __init__(self, intensity):
        self.intensity = intensity


def install_autd_api_stub():
    """実機PCと異なるpyautd3版でも送信ロジックだけを検証する。"""
    pyautd3 = ModuleType("pyautd3")
    pyautd3.AUTD3 = type("AUTD3", (), {"DEVICE_WIDTH": 192.0, "DEVICE_HEIGHT": 151.4})
    pyautd3.FociSTM = type("FociSTM", (), {})
    pyautd3.Hz = 1.0
    pyautd3.Intensity = type("Intensity", (), {"MIN": 0, "MAX": 255})
    pyautd3.OutputMask = type("OutputMask", (), {})
    pyautd3.Static = FakeStatic

    gain = ModuleType("pyautd3.gain")
    holo = ModuleType("pyautd3.gain.holo")
    holo.GSPAT = type("GSPAT", (), {})
    holo.EmissionConstraint = type("EmissionConstraint", (), {})
    holo.GSPATOption = type("GSPATOption", (), {})
    holo.Pa = 1.0

    sys.modules["pyautd3"] = pyautd3
    sys.modules["pyautd3.gain"] = gain
    sys.modules["pyautd3.gain.holo"] = holo


install_autd_api_stub()
sys.modules.pop("core.autd_sender", None)
from core.autd_sender import AutdSender, Static, TargetCommand


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
        self.assertIsInstance(datagram[0], Static)
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
