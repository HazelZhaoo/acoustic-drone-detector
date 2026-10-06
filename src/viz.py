"""Live visual demo: a scrolling spectrogram that turns red when a drone is detected.

Same detector as live.py (0.96 s windows every 0.48 s, 3-of-5 smoothing); this
just draws it. Top: status. Middle: the last few seconds of sound, pitch over
time. Bottom: drone probability over time with the alarm threshold.

Run:
  cd src && ../.venv/bin/python viz.py            # microphone
  cd src && ../.venv/bin/python viz.py clip.wav   # a recording, played through in real time
"""
import queue
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import spectrogram

import audio
from live import Detector

SHOW_S = 8            # seconds of history on screen
FRAME_MS = 50         # redraw interval
DB_RANGE = (-110, -35)  # spectrogram color scale (dB)

BG, INK, MUTED = "#000000", "#ffffff", "#8a8a85"
IDLE, ALARM = "#3987e5", "#e66767"
CMAP_IDLE = LinearSegmentedColormap.from_list("idle", [BG, "#12305c", IDLE, "#cfe2fa"])
CMAP_ALARM = LinearSegmentedColormap.from_list("alarm", [BG, "#5c1414", ALARM, "#ffe1c4"])


class Monitor:
    """Keeps recent audio, runs the detector every hop, remembers the results."""

    def __init__(self, sr: int):
        self.sr = sr
        self.win = int(audio.WINDOW_S * sr)
        self.hop = int(audio.HOP_S * sr)
        self.buffer = np.zeros(0, np.float32)
        self.pending = 0
        self.t = 0.0
        self.history = []  # (time, probability)
        self.p, self.alarm = 0.0, False
        self.detect = Detector()

    def feed(self, chunk: np.ndarray) -> None:
        self.buffer = np.concatenate([self.buffer, chunk])[-SHOW_S * self.sr:]
        self.pending += len(chunk)
        self.t += len(chunk) / self.sr
        while self.pending >= self.hop and self.buffer.size >= self.win:
            self.pending -= self.hop
            window = audio.to_16k(self.buffer[-self.win:], self.sr)[: int(audio.WINDOW_S * audio.SAMPLE_RATE)]
            self.p, self.alarm = self.detect(window)
            self.history.append((self.t, self.p))
        self.history = [(t, p) for t, p in self.history if t > self.t - SHOW_S]

    def spectrogram_db(self) -> np.ndarray:
        x = audio.to_16k(self.buffer, self.sr)
        x = np.pad(x, (SHOW_S * audio.SAMPLE_RATE - x.size, 0))  # left-pad until the screen fills
        _, _, sxx = spectrogram(x, fs=audio.SAMPLE_RATE, nperseg=512, noverlap=352)
        return 10 * np.log10(sxx + 1e-14)


class View:
    def __init__(self, monitor: Monitor):
        self.m = monitor
        plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "text.color": INK,
                             "axes.edgecolor": "#2a2a28", "xtick.color": MUTED, "ytick.color": MUTED,
                             "axes.labelcolor": MUTED, "font.size": 12})
        self.fig = plt.figure(figsize=(11, 6.5))
        self.fig.canvas.manager.set_window_title("Acoustic Drone Detector")
        grid = self.fig.add_gridspec(3, 1, height_ratios=[0.5, 3, 1.1], hspace=0.25,
                                     left=0.07, right=0.97, top=0.97, bottom=0.08)

        top = self.fig.add_subplot(grid[0])
        top.axis("off")
        self.status = top.text(0, 0.5, "", fontsize=26, fontweight="bold", va="center")
        self.prob_text = top.text(1, 0.5, "", fontsize=18, ha="right", va="center", color=MUTED)

        self.spec_ax = self.fig.add_subplot(grid[1])
        self.image = self.spec_ax.imshow(
            self.m.spectrogram_db(), origin="lower", aspect="auto", cmap=CMAP_IDLE,
            extent=[-SHOW_S, 0, 0, audio.SAMPLE_RATE / 2000], vmin=DB_RANGE[0], vmax=DB_RANGE[1])
        self.spec_ax.set_ylabel("Pitch (kHz)")
        self.spec_ax.tick_params(labelbottom=False)

        self.prob_ax = self.fig.add_subplot(grid[2], sharex=self.spec_ax)
        self.prob_ax.axhline(self.m.detect.threshold, color=MUTED, linestyle="--", linewidth=1)
        self.prob_ax.text(-SHOW_S + 0.1, self.m.detect.threshold + 0.05, "alarm threshold",
                          color=MUTED, fontsize=10)
        (self.line,) = self.prob_ax.plot([], [], linewidth=2.5, color=IDLE)
        self.prob_ax.set_xlim(-SHOW_S, 0)
        self.prob_ax.set_ylim(-0.05, 1.05)
        self.prob_ax.set_yticks([0, 0.5, 1])
        self.prob_ax.set_ylabel("Drone prob.")
        self.prob_ax.set_xlabel("Seconds ago")
        self.draw()

    def draw(self) -> None:
        color = ALARM if self.m.alarm else IDLE
        self.image.set_data(self.m.spectrogram_db())
        self.image.set_cmap(CMAP_ALARM if self.m.alarm else CMAP_IDLE)
        self.status.set_text("● DRONE DETECTED" if self.m.alarm else "● LISTENING")
        self.status.set_color(color)
        self.prob_text.set_text(f"drone probability {self.m.p:.2f}")
        if self.m.history:
            t, p = zip(*self.m.history)
            self.line.set_data(np.array(t) - self.m.t, p)
        self.line.set_color(color)


def run(source: str | None) -> None:
    chunks = queue.Queue()
    if source:  # recording: feed it in real-time-sized chunks
        x = audio.load(source)
        sr = audio.SAMPLE_RATE
        step = int(sr * FRAME_MS / 1000)
        for i in range(0, x.size, step):
            chunks.put(x[i : i + step])
        stream = None
    else:
        import sounddevice as sd

        sr = int(sd.query_devices(kind="input")["default_samplerate"])
        stream = sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=sr // 20,
                                callback=lambda data, *_: chunks.put(data[:, 0].copy()))

    view = View(Monitor(sr))

    def update(_):
        if source:  # one chunk per frame = real time
            if not chunks.empty():
                view.m.feed(chunks.get())
        else:
            while not chunks.empty():
                view.m.feed(chunks.get())
        view.draw()

    anim = FuncAnimation(view.fig, update, interval=FRAME_MS, cache_frame_data=False)  # noqa: F841
    if stream:
        stream.start()
    plt.show()
    if stream:
        stream.stop()


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
