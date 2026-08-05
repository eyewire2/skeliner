"""Tests for skeliner.plot.viewer — the interactive server.

The viewer mirrors an operation's terminal output into the browser over
the WebSocket, which is the only sign a long-running step is moving.
"""

import asyncio
import contextlib
import io
import threading

from skeliner.plot.viewer import _LogTee
from skeliner.skeletonize import _timed


def _tee_output(emit):
    """Run *emit* through a ``_LogTee``.

    Returns what it broadcast and what reached the wrapped stream.
    """
    sent = []
    original = io.StringIO()

    async def broadcast(msg):
        sent.append(msg["text"])

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        tee = _LogTee(original, loop, broadcast)
        with contextlib.redirect_stdout(tee):
            emit()
        tee.finish()
        # queued in order, so this one completing means all of them did
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop).result(5)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()
    return sent, original.getvalue()


# ── _LogTee ───────────────────────────────────────────────────────────
#
# Driven through the real ``_timed`` helper rather than hand-written
# strings, so these cannot drift from what the pipeline actually prints.


def test_timed_stage_reaches_the_browser_as_one_line():
    """A stage is a label printed before the work and an elapsed time
    printed after, on the same line.  The browser must get the label
    immediately — it is the only sign a long stage is running — and then
    the finished line, never a bare "1.99 s" on its own."""

    def emit():
        with _timed("↳  build surface graph", verbose=True):
            pass

    sent, _ = _tee_output(emit)
    assert len(sent) == 2, sent
    assert sent[0].strip().startswith("↳  build surface graph")
    assert sent[0].endswith("…"), "the label must arrive before the timing"
    assert sent[1].startswith(sent[0]), "the finished line completes the label"
    assert sent[1].endswith(" s")


def test_timed_sub_messages_are_their_own_lines():
    def emit():
        with _timed("↳  skeletonize neurites", verbose=True) as log:
            log("neurite 0: 12 verts → 3 nodes")

    sent, _ = _tee_output(emit)
    assert len(sent) == 3, sent
    assert sent[2].strip() == "└─ neurite 0: 12 verts → 3 nodes"


def test_whole_lines_are_sent_once_each_and_reach_the_terminal():
    def emit():
        print("alpha")
        print("beta")

    sent, raw = _tee_output(emit)
    assert sent == ["alpha", "beta"]
    assert raw == "alpha\nbeta\n", "the terminal must still get everything"
