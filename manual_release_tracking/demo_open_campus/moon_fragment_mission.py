from __future__ import annotations

import argparse
import csv
import math
import random
import time
import tkinter as tk
from pathlib import Path


BACKGROUND = "#050821"
CYAN = "#57e6ff"
GOLD = "#ffd166"
WHITE = "#f7fbff"
GREEN = "#66f2a4"

STORY_DISCOVERY_SEC = 5.0
STORY_LOADING_SEC = 5.0
STORY_ACCIDENT_SEC = 5.0


class CaptureEventTailer:
    """Read only newly appended capture-event rows from a CSV file."""

    def __init__(self, path: Path):
        self.path = path
        self._header: list[str] | None = None
        self._offset = 0
        self._remainder = ""
        self._prime()

    def _prime(self):
        if not self.path.is_file():
            return
        with self.path.open("r", encoding="utf-8-sig", newline="") as stream:
            header_line = stream.readline()
            if header_line:
                self._header = next(csv.reader([header_line]))
            stream.seek(0, 2)
            self._offset = stream.tell()

    def poll(self) -> list[dict[str, str]]:
        if not self.path.is_file():
            return []
        size = self.path.stat().st_size
        if size < self._offset:
            self._header = None
            self._offset = 0
            self._remainder = ""
        with self.path.open("r", encoding="utf-8-sig", newline="") as stream:
            if self._header is None:
                header_line = stream.readline()
                if not header_line:
                    return []
                self._header = next(csv.reader([header_line]))
                self._offset = stream.tell()
            stream.seek(self._offset)
            chunk = stream.read()
            self._offset = stream.tell()
        if not chunk:
            return []
        text = self._remainder + chunk
        lines = text.splitlines(keepends=True)
        self._remainder = ""
        if lines and not lines[-1].endswith(("\n", "\r")):
            self._remainder = lines.pop()
        rows = []
        for line in lines:
            if not line.strip():
                continue
            rows.extend(csv.DictReader([line], fieldnames=self._header))
        return rows


class MoonFragmentMission:
    def __init__(
        self,
        root: tk.Tk,
        *,
        events_csv: Path,
        success_hold_sec: float,
        failure_timeout_sec: float,
        fullscreen: bool,
    ):
        self.root = root
        self.root.title("月のかけら救出ミッション")
        self.root.configure(bg=BACKGROUND)
        self.root.attributes("-fullscreen", fullscreen)
        self.canvas = tk.Canvas(root, bg=BACKGROUND, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.tailer = CaptureEventTailer(events_csv)
        self.success_hold_sec = float(success_hold_sec)
        self.failure_timeout_sec = float(failure_timeout_sec)
        self.state = "INTRO"
        self.state_started_at = time.monotonic()
        self.release_started_at: float | None = None
        self.local_hold_reached = False
        self.failure_reason = ""
        self.stars = [
            (random.random(), random.random(), random.uniform(1.0, 3.2), random.random() * 6.28)
            for _ in range(95)
        ]
        self.fireworks = []
        self.root.bind("<space>", self._on_space)
        self.root.bind("<Return>", lambda _event: self._simulate_release())
        self.root.bind("<Key-s>", lambda _event: self._set_state("SUCCESS"))
        self.root.bind("<Key-f>", lambda _event: self._fail("operator_test"))
        self.root.bind("<Key-r>", lambda _event: self._set_state("READY"))
        self.root.bind("<Key-0>", lambda _event: self._set_state("INTRO"))
        self.root.bind("<KP_0>", lambda _event: self._set_state("INTRO"))
        self.root.bind("<Escape>", lambda _event: self.root.destroy())
        self.root.after(33, self._tick)

    def _set_state(self, state: str):
        self.state = state
        self.state_started_at = time.monotonic()
        if state in {"INTRO", "READY", "COUNTDOWN", "WAIT_RELEASE"}:
            self.release_started_at = None
            self.local_hold_reached = False
            self.failure_reason = ""
        if state == "SUCCESS":
            self._make_fireworks()

    def _on_space(self, _event):
        if self.state == "INTRO":
            self._set_state("STORY_DISCOVERY")
        elif self.state.startswith("STORY_"):
            self._set_state("READY")
        elif self.state == "READY":
            self._set_state("COUNTDOWN")
        elif self.state in {"SUCCESS", "FAILURE"}:
            self._set_state("READY")

    def _simulate_release(self):
        if self.state in {"COUNTDOWN", "WAIT_RELEASE", "READY"}:
            self.release_started_at = time.monotonic()
            self.local_hold_reached = True
            self._set_state("NET")
            self.release_started_at = time.monotonic()

    def _fail(self, reason: str):
        if self.state == "SUCCESS":
            return
        self.failure_reason = reason
        self._set_state("FAILURE")

    def _consume_capture_events(self):
        for row in self.tailer.poll():
            event = row.get("event", "")
            if event == "release_candidate_command_enqueued":
                if self.state in {"READY", "COUNTDOWN", "WAIT_RELEASE", "NET"}:
                    self._set_state("NET")
                    self.release_started_at = time.monotonic()
                    self.local_hold_reached = False
            elif event == "capture_braking_to_local_hold" and self.state == "NET":
                self.local_hold_reached = True
            elif event == "automatic_field_stopped" and self.state == "NET":
                self._fail(row.get("reason", "capture_stopped"))

    def _make_fireworks(self):
        palette = [GOLD, CYAN, GREEN, "#ff8bd8", "#ff8c5a", WHITE]
        self.fireworks = []
        for _ in range(5):
            cx = random.uniform(0.18, 0.82)
            cy = random.uniform(0.12, 0.48)
            color = random.choice(palette)
            phase = random.uniform(0.0, 1.2)
            self.fireworks.append((cx, cy, color, phase))

    def _tick(self):
        now = time.monotonic()
        self._consume_capture_events()
        if (
            self.state == "STORY_DISCOVERY"
            and now - self.state_started_at >= STORY_DISCOVERY_SEC
        ):
            self._set_state("STORY_LOADING")
        elif (
            self.state == "STORY_LOADING"
            and now - self.state_started_at >= STORY_LOADING_SEC
        ):
            self._set_state("STORY_ACCIDENT")
        elif (
            self.state == "STORY_ACCIDENT"
            and now - self.state_started_at >= STORY_ACCIDENT_SEC
        ):
            self._set_state("READY")
        if self.state == "COUNTDOWN" and now - self.state_started_at >= 3.6:
            self._set_state("WAIT_RELEASE")
        if (
            self.state == "NET"
            and self.release_started_at is not None
            and self.local_hold_reached
            and now - self.release_started_at >= self.success_hold_sec
        ):
            self._set_state("SUCCESS")
        elif (
            self.state == "NET"
            and self.release_started_at is not None
            and now - self.release_started_at >= self.failure_timeout_sec
        ):
            self._fail("capture_timeout")
        self._draw(now)
        self.root.after(33, self._tick)

    def _text(self, x, y, text, *, size, color=WHITE, weight="bold", anchor="center"):
        self.canvas.create_text(
            x,
            y,
            text=text,
            fill=color,
            font=("Yu Gothic UI", size, weight),
            anchor=anchor,
            justify="center",
        )

    def _draw_space(self, width: int, height: int, now: float):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, width, height, fill=BACKGROUND, outline="")
        for x, y, radius, phase in self.stars:
            pulse = 0.55 + 0.45 * math.sin(now * 1.7 + phase)
            color = WHITE if pulse > 0.72 else "#7f91c9"
            r = radius * (0.7 + 0.3 * pulse)
            self.canvas.create_oval(
                x * width - r,
                y * height - r,
                x * width + r,
                y * height + r,
                fill=color,
                outline="",
            )

    def _draw_spaceship(self, x: float, y: float, scale: float = 1.0, flame=False):
        c = self.canvas
        c.create_polygon(
            x - 95 * scale, y,
            x - 35 * scale, y - 45 * scale,
            x + 70 * scale, y - 25 * scale,
            x + 100 * scale, y,
            x + 70 * scale, y + 25 * scale,
            x - 35 * scale, y + 45 * scale,
            fill="#d8e6ff", outline=CYAN, width=3,
        )
        c.create_oval(
            x - 15 * scale, y - 35 * scale,
            x + 40 * scale, y + 5 * scale,
            fill="#3b5ca8", outline=WHITE, width=2,
        )
        c.create_polygon(
            x - 55 * scale, y - 35 * scale,
            x - 90 * scale, y - 70 * scale,
            x - 15 * scale, y - 40 * scale,
            fill="#7d91c9", outline=CYAN,
        )
        if flame:
            length = 65 * scale * (0.85 + 0.15 * math.sin(time.monotonic() * 18))
            c.create_polygon(
                x - 95 * scale, y - 18 * scale,
                x - (95 * scale + length), y,
                x - 95 * scale, y + 18 * scale,
                fill=GOLD, outline="#ff7b54", width=2,
            )

    def _draw_fragment(self, x: float, y: float, radius: float):
        c = self.canvas
        c.create_oval(x - radius, y - radius, x + radius, y + radius,
                      fill="#e9edf7", outline=CYAN, width=4)
        c.create_oval(x - radius * 0.48, y - radius * 0.35,
                      x - radius * 0.05, y + radius * 0.08,
                      fill="#c5ccdc", outline="")
        c.create_oval(x + radius * 0.18, y + radius * 0.05,
                      x + radius * 0.55, y + radius * 0.42,
                      fill="#b8c1d5", outline="")

    def _draw_robot(self, x: float, y: float, scale: float = 1.0):
        c = self.canvas
        c.create_rectangle(x - 48 * scale, y - 48 * scale,
                           x + 48 * scale, y + 35 * scale,
                           fill="#b9c7e8", outline=CYAN, width=3)
        c.create_oval(x - 28 * scale, y - 22 * scale,
                      x - 12 * scale, y - 6 * scale, fill=CYAN, outline="")
        c.create_oval(x + 12 * scale, y - 22 * scale,
                      x + 28 * scale, y - 6 * scale, fill=CYAN, outline="")
        c.create_line(x, y - 48 * scale, x, y - 75 * scale, fill=CYAN, width=3)
        c.create_oval(x - 7 * scale, y - 85 * scale,
                      x + 7 * scale, y - 71 * scale, fill=GOLD, outline="")
        c.create_line(x - 48 * scale, y, x - 85 * scale, y - 30 * scale,
                      fill="#b9c7e8", width=10)
        c.create_line(x + 48 * scale, y, x + 85 * scale, y - 30 * scale,
                      fill="#b9c7e8", width=10)

    def _draw_moon_surface(self, width: int, height: int):
        horizon = height * 0.72
        self.canvas.create_polygon(
            0, horizon,
            width * 0.18, horizon - 28,
            width * 0.36, horizon + 5,
            width * 0.55, horizon - 22,
            width * 0.76, horizon + 2,
            width, horizon - 35,
            width, height,
            0, height,
            fill="#858ba4",
            outline="#cdd5e8",
            width=2,
        )
        for cx, cy, radius in (
            (0.14, 0.84, 55),
            (0.38, 0.78, 32),
            (0.68, 0.88, 68),
            (0.88, 0.77, 38),
        ):
            self.canvas.create_oval(
                cx * width - radius,
                cy * height - radius * 0.35,
                cx * width + radius,
                cy * height + radius * 0.35,
                fill="#696f87",
                outline="#aeb7ce",
                width=2,
            )

    def _draw_fireworks(self, width: int, height: int, elapsed: float):
        for cx, cy, color, phase in self.fireworks:
            age = max(0.0, (elapsed - phase) % 1.8)
            radius = min(150.0, age * 115.0)
            alpha_phase = max(0.0, 1.0 - age / 1.8)
            if alpha_phase <= 0.05:
                continue
            for index in range(16):
                angle = index * math.tau / 16.0
                x = cx * width + math.cos(angle) * radius
                y = cy * height + math.sin(angle) * radius
                r = 2.5 + 3.5 * alpha_phase
                self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                        fill=color, outline="")

    def _draw(self, now: float):
        width = max(960, self.canvas.winfo_width())
        height = max(600, self.canvas.winfo_height())
        self._draw_space(width, height, now)
        elapsed = now - self.state_started_at

        if self.state == "INTRO":
            self._draw_moon_surface(width, height)
            self._draw_spaceship(width * 0.74, height * 0.31, 0.82)
            self._draw_fragment(width * 0.34, height * 0.67, 38)
            self._text(width * 0.5, height * 0.11, "月のかけら救出ミッション", size=42, color=GOLD)
            self._text(
                width * 0.5,
                height * 0.49,
                "20XX年、月面探査隊は\n"
                "ふしぎに光る石を発見しました",
                size=27,
            )
            self._text(width * 0.5, height * 0.92, "SPACEキーで物語をスタート", size=19, color=CYAN)

        elif self.state == "STORY_DISCOVERY":
            self._draw_moon_surface(width, height)
            discovery_progress = min(1.0, elapsed / (STORY_DISCOVERY_SEC * 0.8))
            robot_x = width * (0.24 + 0.14 * discovery_progress)
            self._draw_robot(robot_x, height * 0.67, 0.72)
            glow = 42 + 7 * math.sin(now * 5.0)
            self.canvas.create_oval(
                width * 0.57 - glow,
                height * 0.68 - glow,
                width * 0.57 + glow,
                height * 0.68 + glow,
                outline=GOLD,
                width=3,
            )
            self._draw_fragment(width * 0.57, height * 0.68, 33)
            self._text(width * 0.5, height * 0.13, "月面探査隊が光る石を発見！", size=38, color=GOLD)
            self._text(width * 0.5, height * 0.27, "これは貴重な『月のかけら』かもしれない", size=25)
            self._text(width * 0.92, height * 0.94, "SPACEでスキップ", size=14, color=CYAN, anchor="e")

        elif self.state == "STORY_LOADING":
            self._draw_moon_surface(width, height)
            self._draw_spaceship(width * 0.73, height * 0.39, 0.85)
            progress = min(1.0, elapsed / (STORY_LOADING_SEC * 0.8))
            fragment_x = width * (0.38 + 0.31 * progress)
            fragment_y = height * (0.69 - 0.28 * progress)
            self._draw_fragment(fragment_x, fragment_y, 32)
            self._draw_robot(width * 0.34, height * 0.68, 0.68)
            self._text(width * 0.5, height * 0.13, "地球の研究所へ届けよう", size=39, color=GREEN)
            self._text(width * 0.5, height * 0.25, "月のかけらを宇宙船に積み込みます", size=25)
            self._text(width * 0.92, height * 0.94, "SPACEでスキップ", size=14, color=CYAN, anchor="e")

        elif self.state == "STORY_ACCIDENT":
            shake = math.sin(now * 22.0) * min(12.0, elapsed * 10.0)
            ship_x = width * 0.57 + shake
            ship_y = height * 0.39 + math.sin(now * 17.0) * 5
            self._draw_spaceship(ship_x, ship_y, 0.92, flame=True)
            accident_progress = min(1.0, elapsed / (STORY_ACCIDENT_SEC * 0.8))
            fall = height * 0.27 * accident_progress
            self._draw_fragment(ship_x + 55, ship_y + 68 + fall, 35)
            if int(elapsed * 4) % 2 == 0:
                self._text(width * 0.5, height * 0.12, "！緊急事態！", size=44, color="#ff7891")
            self._text(
                width * 0.5,
                height * 0.72,
                "宇宙船が大きく揺れて、月のかけらが落ちそう！\n"
                "見えない超音波ネットで助けよう",
                size=27,
            )
            self._text(width * 0.92, height * 0.94, "SPACEでスキップ", size=14, color=CYAN, anchor="e")

        elif self.state == "READY":
            self._draw_spaceship(width * 0.73, height * 0.25, 0.75)
            self._draw_fragment(width * 0.5, height * 0.48 + math.sin(now * 2) * 5, 55)
            self._text(width * 0.5, height * 0.13, "あなたの出番！ 救出準備完了", size=42, color=GREEN)
            self._text(width * 0.5, height * 0.68, "球体を持って、合図を待ってね", size=27)
            self._text(width * 0.5, height * 0.77, "手を離すまで超音波は OFF", size=24, color=CYAN)
            self._text(width * 0.5, height * 0.91, "SPACEキーでカウントダウン", size=18, color=GOLD)

        elif self.state == "COUNTDOWN":
            number = 3 - int(elapsed / 1.0)
            if number >= 1:
                self._text(width * 0.5, height * 0.44, str(number), size=120, color=GOLD)
                self._text(width * 0.5, height * 0.68, "月のかけらをそっと離そう", size=28)
            else:
                self._text(width * 0.5, height * 0.43, "リリース！", size=78, color=GREEN)
                self._text(width * 0.5, height * 0.68, "手を離して！", size=34)
            self._draw_fragment(width * 0.5, height * 0.80, 44)

        elif self.state == "WAIT_RELEASE":
            self._draw_fragment(width * 0.5, height * 0.50 + min(45, elapsed * 22), 48)
            self._text(width * 0.5, height * 0.19, "リリース！", size=72, color=GOLD)
            self._text(width * 0.5, height * 0.76, "解放を検出中……", size=28, color=CYAN)

        elif self.state == "NET":
            drop = min(115.0, elapsed * 170.0)
            self._draw_fragment(width * 0.5, height * 0.38 + drop, 50)
            for ring in range(3):
                radius = 85 + ((elapsed * 130 + ring * 55) % 180)
                self.canvas.create_oval(
                    width * 0.5 - radius,
                    height * 0.53 - radius * 0.26,
                    width * 0.5 + radius,
                    height * 0.53 + radius * 0.26,
                    outline=CYAN,
                    width=3,
                )
            self._text(width * 0.5, height * 0.15, "超音波ネット展開中……", size=42, color=CYAN)
            progress = 0.0
            if self.release_started_at is not None:
                progress = min(1.0, (now - self.release_started_at) / self.success_hold_sec)
            x0, x1, y = width * 0.28, width * 0.72, height * 0.86
            self.canvas.create_rectangle(x0, y, x1, y + 20, outline=WHITE, width=2)
            self.canvas.create_rectangle(x0, y, x0 + (x1 - x0) * progress, y + 20,
                                         fill=GREEN, outline="")

        elif self.state == "SUCCESS":
            launch_x = width * 0.63 + min(width * 0.55, elapsed * width * 0.13)
            self._draw_fireworks(width, height, elapsed)
            self._draw_spaceship(launch_x, height * 0.65 - min(180, elapsed * 35), 0.82, flame=True)
            self._draw_fragment(launch_x + 10, height * 0.65 - min(180, elapsed * 35), 25)
            self._text(width * 0.5, height * 0.17, "救出成功！", size=66, color=GREEN)
            self._text(width * 0.5, height * 0.29, "月のかけらを取り戻した！", size=35, color=GOLD)
            self._text(width * 0.5, height * 0.92, "SPACEキーで次のミッション", size=18, color=CYAN)

        elif self.state == "FAILURE":
            robot_x = min(width * 0.50, width * 0.12 + elapsed * width * 0.18)
            self._draw_fragment(width * 0.50, height * 0.68, 43)
            self._draw_robot(robot_x, height * 0.68, 0.85)
            self._text(width * 0.5, height * 0.15, "救出データを取得したよ！", size=42, color=GOLD)
            self._text(width * 0.5, height * 0.28, "ロボットが月のかけらを回収中", size=28)
            self._text(width * 0.5, height * 0.39, "もう一度チャレンジしよう", size=34, color=GREEN)
            self._text(width * 0.5, height * 0.92, "SPACEキーでもう一度", size=18, color=CYAN)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Children's moon-fragment rescue display for the capture demo."
    )
    parser.add_argument(
        "--events-csv",
        type=Path,
        default=Path("./manual_release_tracking/auto_release_capture_events.csv"),
        help="Live auto_release_capture_events.csv written by the controller.",
    )
    parser.add_argument(
        "--success-hold-sec",
        type=float,
        default=2.0,
        help="Public-demo hold duration required before the success scene.",
    )
    parser.add_argument(
        "--failure-timeout-sec",
        type=float,
        default=4.0,
        help="Show the recovery scene if capture never reaches hold by this time.",
    )
    parser.add_argument("--fullscreen", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.success_hold_sec <= 0.0 or args.failure_timeout_sec <= 0.0:
        raise SystemExit("hold and timeout durations must be positive")
    if args.failure_timeout_sec <= args.success_hold_sec:
        raise SystemExit("--failure-timeout-sec must exceed --success-hold-sec")
    root = tk.Tk()
    MoonFragmentMission(
        root,
        events_csv=args.events_csv,
        success_hold_sec=args.success_hold_sec,
        failure_timeout_sec=args.failure_timeout_sec,
        fullscreen=args.fullscreen,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
