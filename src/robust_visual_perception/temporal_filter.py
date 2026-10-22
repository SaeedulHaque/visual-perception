"""N-of-K persistence filter for the per-frame safety state.

STOP is asserted when at least ``n_on`` of the last ``k_window`` valid frames
were STOP. Once asserted, it is released only after ``m_off`` consecutive
CLEAR frames. This trades a few frames of delay for fewer flickering state
changes; the experiments measure both sides of that trade.
"""

from __future__ import annotations

from collections import deque

from robust_visual_perception.decision import State


class PersistenceFilter:
    def __init__(self, n_on: int = 3, k_window: int = 5, m_off: int = 5) -> None:
        if not 1 <= n_on <= k_window:
            raise ValueError(f"need 1 <= n_on <= k_window, got {n_on}, {k_window}")
        if m_off < 1:
            raise ValueError(f"m_off must be >= 1, got {m_off}")
        self.n_on = n_on
        self.k_window = k_window
        self.m_off = m_off
        self.reset()

    def reset(self) -> None:
        """Forget everything; start CLEAR, as at the beginning of a stream."""
        self.state = State.CLEAR
        self._window: deque[State] = deque(maxlen=self.k_window)
        self._clear_run = 0

    def update(self, observed: State) -> State:
        """Feed one per-frame decision and return the filtered state."""
        if observed is State.UNKNOWN:
            # No evidence this frame: report it, and start collecting fresh
            # evidence afterwards. The asserted state is kept, so an active
            # STOP still needs m_off new CLEAR frames to release.
            self._window.clear()
            self._clear_run = 0
            return State.UNKNOWN

        self._window.append(observed)
        if self.state is State.CLEAR:
            if sum(s is State.STOP for s in self._window) >= self.n_on:
                self.state = State.STOP
                self._clear_run = 0
        else:
            self._clear_run = self._clear_run + 1 if observed is State.CLEAR else 0
            if self._clear_run >= self.m_off:
                self.state = State.CLEAR
                self._clear_run = 0
                # Old STOP frames must not re-trigger right after release.
                self._window.clear()
        return self.state
