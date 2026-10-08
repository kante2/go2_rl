"""Bound normal velocity/position target changes without relaxing fault limits."""
import numpy as np


class VelocityRamp:
    def __init__(self, acceleration, deceleration, dt):
        self.acceleration = np.asarray(acceleration, dtype=float)
        self.deceleration = np.asarray(deceleration, dtype=float)
        self.dt = float(dt)
        if (self.acceleration.shape != (3,) or self.deceleration.shape != (3,)
                or not np.isfinite([*self.acceleration, *self.deceleration, self.dt]).all()
                or min(*self.acceleration, *self.deceleration, self.dt) <= 0):
            raise ValueError('Command ramp rates and period must be finite and positive')
        self.value = np.zeros(3)

    def reset(self):
        self.value[:] = 0

    def step(self, requested):
        requested = np.asarray(requested, dtype=float)
        if requested.shape != (3,) or not np.isfinite(requested).all():
            raise ValueError('Invalid requested velocity')
        reversing = self.value * requested < 0
        goal = np.where(reversing, 0., requested)
        slowing = np.abs(goal) < np.abs(self.value)
        limit = np.where(slowing, self.deceleration, self.acceleration) * self.dt
        self.value += np.clip(goal - self.value, -limit, limit)
        # Snap floating-point residues to zero before accelerating in reverse.
        self.value[np.abs(self.value) < 1e-10] = 0
        return self.value.copy()


class JointTargetRamp:
    def __init__(self, max_rate, dt, *, transition_rate=None, transition_seconds=0.):
        self.dt = float(dt)
        self.step_limit = float(max_rate) * self.dt
        if not np.isfinite(self.step_limit) or self.step_limit <= 0:
            raise ValueError('Joint target rate and period must be finite and positive')
        self.transition_limit = self.step_limit if transition_rate is None else float(transition_rate) * self.dt
        self.transition_seconds = float(transition_seconds)
        if (not np.isfinite([self.transition_limit, self.transition_seconds]).all()
                or not 0 < self.transition_limit <= self.step_limit or self.transition_seconds < 0):
            raise ValueError('Invalid gait transition rate or duration')
        self.value = None
        self.moving = None
        self.transition_remaining = 0.

    def step(self, requested, initial=None, *, moving=None):
        requested = np.asarray(requested, dtype=np.float32)
        if requested.shape != (12,) or not np.isfinite(requested).all():
            raise ValueError('Invalid joint target')
        if initial is not None:
            initial = np.asarray(initial, dtype=np.float32)
            if initial.shape != (12,) or not np.isfinite(initial).all():
                raise ValueError('Invalid initial joint target')
            self.value = initial.copy()
            self.moving = None
            self.transition_remaining = 0.
        if self.value is None:
            raise ValueError('Initialize joint target ramp from measured joints')
        # The trained observation abruptly zeros gait phase below |cmd|=0.1.
        # Keep that observation contract, but ease actuator targets across the
        # walking-to-standing boundary. Never clamp against measured tracking
        # error: a stuck joint must still trip the unchanged tracking guard.
        if moving is not None:
            if self.moving is True and not moving:
                self.transition_remaining = self.transition_seconds
            self.moving = bool(moving)
        limit = self.transition_limit if self.transition_remaining > 1e-9 else self.step_limit
        self.value += np.clip(requested - self.value, -limit, limit)
        self.transition_remaining = max(0., self.transition_remaining - self.dt)
        return self.value.copy()
