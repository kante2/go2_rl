"""Physics-substep measurements, independent of ROS, MuJoCo and the UI."""

FEET = ("FR", "FL", "RR", "RL")
LANDING_WINDOW = 0.050
CONTACT_THRESHOLD = 1.0
MIN_AIR_TIME = 0.020


class FootMetrics:
    """Integrate vertical force for 50 ms after a qualified touchdown.

    Standing support is not a new landing. Require 20 ms airborne before
    touchdown; do not count initially planted feet or contact chatter.
    An early liftoff completes the integral at that time. Force is |Fz| in
    world coordinates, appropriate for the flat ground used by these tasks.
    """

    def __init__(self):
        self.time = 0.0
        self.feet = [dict(contact=False, force_n=0.0, peak_n=0.0, air_s=0.0,
                          remaining_s=0.0, integral=0.0, landing_peak=0.0,
                          landing_impulse_ns=None, landing_peak_n=None,
                          landing_time_s=None, landing_count=0) for _ in FEET]

    def sample(self, vertical_forces, dt):
        if dt <= 0:
            raise ValueError("physics dt must be positive")
        self.time += dt
        for foot, signed_force in zip(self.feet, vertical_forces):
            force = abs(float(signed_force))
            contact = force > CONTACT_THRESHOLD
            touchdown = contact and not foot["contact"] and foot["air_s"] >= MIN_AIR_TIME - 1e-9
            if touchdown:
                foot.update(remaining_s=LANDING_WINDOW, integral=0.0, landing_peak=0.0)
            if foot["remaining_s"] > 0:
                if contact:
                    foot["integral"] += force * min(dt, foot["remaining_s"])
                    foot["landing_peak"] = max(foot["landing_peak"], force)
                    foot["remaining_s"] = max(0.0, foot["remaining_s"] - dt)
                if not contact or foot["remaining_s"] < 1e-9:
                    foot.update(landing_impulse_ns=foot["integral"],
                                landing_peak_n=foot["landing_peak"], landing_time_s=self.time,
                                remaining_s=0.0, landing_count=foot["landing_count"] + 1)
            foot["air_s"] = 0.0 if contact else foot["air_s"] + dt
            foot["contact"] = contact
            foot["force_n"] = force
            foot["peak_n"] = max(foot["peak_n"], force)

    def snapshot(self):
        fields = ("contact", "force_n", "peak_n", "landing_impulse_ns",
                  "landing_peak_n", "landing_time_s", "landing_count")
        return {key: [foot[key] for foot in self.feet] for key in fields}

    def clear_interval_peak(self):
        for foot in self.feet:
            foot["peak_n"] = 0.0
