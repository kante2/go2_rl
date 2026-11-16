"""CPU MuJoCo low-command rollout; no policy inference and no robot connection."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from real2sim_common import DEFAULT_MODEL, JOINTS, LEGS, command_at, rotation

DEFAULT_PARAMETERS = {"gain_hip": 1., "gain_thigh": 1., "gain_calf": 1.,
                      "damping_hip": 1., "damping_thigh": 1., "damping_calf": 1.,
                      "delay_s": 0., "base_mass_scale": 1.}
# Normalizations are explicit physical scales, fixed before fitting.
LOSS_SCALES = {"q": .1, "dq": 1., "gyro": .3, "orientation": .1}


def parameters_from_file(path):
    if path is None:
        return DEFAULT_PARAMETERS.copy()
    value = json.loads(Path(path).read_text())
    value = value.get("parameters", value)
    unknown = set(value) - set(DEFAULT_PARAMETERS)
    if unknown:
        raise ValueError(f"Unknown parameter keys: {sorted(unknown)}")
    result = DEFAULT_PARAMETERS | value
    if not all(np.isfinite(v) for v in result.values()) or not 0 <= result["delay_s"] <= .2:
        raise ValueError("Invalid parameters/delay")
    if any(v <= 0 for k, v in result.items() if k != "delay_s"):
        raise ValueError("Gains, damping and mass scale must be positive")
    return result


class Replay:
    def __init__(self, model_path=DEFAULT_MODEL, timestep=.001, height_offset=0.):
        import mujoco
        self.mj = mujoco
        self.model_path = Path(model_path).resolve()
        if not np.isfinite(timestep) or timestep <= 0 or not np.isfinite(height_offset):
            raise ValueError("Invalid timestep/height offset")
        tree = ET.parse(model_path)
        root = tree.getroot()
        compiler = root.find("compiler")
        compiler.set("meshdir", str((Path(model_path).parent / compiler.get("meshdir", ".")).resolve()))
        option = root.find("option")
        if option is None:
            option = ET.SubElement(root, "option")
        option.set("timestep", str(timestep))
        option.set("integrator", "implicitfast")
        option.set("cone", "elliptic")
        option.set("gravity", "0 0 -9.81")
        world = root.find("worldbody")
        ET.SubElement(world, "geom", name="real2sim_floor", type="plane", size="0 0 .05",
                      contype="0", conaffinity="1", friction=".6 .005 .0001")
        # Explicitly reproduce FULL_COLLISION and armature from go2_constants.py.
        for geom in world.iter("geom"):
            if "collision" in geom.get("name", ""):
                geom.set("contype", "1")
                geom.set("conaffinity", "0")
                geom.set("condim", "3" if "foot" in geom.get("name") else "1")
                if "foot" in geom.get("name"):
                    geom.set("friction", ".6 .005 .0001")
                    geom.set("solimp", ".9 .95 .023 .5 2")
        for joint in world.iter("joint"):
            if joint.get("name") in JOINTS:
                joint.set("armature", ".02" if "calf" in joint.get("name") else ".01")
                joint.set("damping", "0")
                joint.set("frictionloss", "0")
        actuator = ET.SubElement(root, "actuator")
        for name in JOINTS:
            limit = 45. if "calf" in name else 23.5
            ET.SubElement(actuator, "motor", name="replay_" + name, joint=name,
                          gear="1", ctrllimited="true", ctrlrange=f"{-limit} {limit}")
        self.xml = ET.tostring(root, encoding="unicode")
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.qids = np.array([m.jnt_qposadr[m.joint(n).id] for n in JOINTS])
        self.dids = np.array([m.jnt_dofadr[m.joint(n).id] for n in JOINTS])
        self.aids = np.array([m.actuator("replay_" + n).id for n in JOINTS])
        self.feet = np.array([m.geom(leg + "_foot_collision").id for leg in LEGS])
        self.floor = m.geom("real2sim_floor").id
        self.base = m.body("base_link").id
        self.base_mass = float(m.body_mass[self.base])
        self.base_inertia = m.body_inertia[self.base].copy()
        self.limits = np.tile([23.5, 23.5, 45.], 4)
        self.height_offset = height_offset
        self.timestep = timestep

    def initialize(self, dataset, index, parameters):
        m, d, mj = self.model, self.data, self.mj
        # Preserve physical inertia by scaling mass AND inertia together.
        m.body_mass[self.base] = self.base_mass * parameters["base_mass_scale"]
        m.body_inertia[self.base] = self.base_inertia * parameters["base_mass_scale"]
        mj.mj_setConst(m, d)
        mj.mj_resetData(m, d)
        d.qpos[:3] = 0.
        d.qpos[3:7] = dataset["quaternion"][index] / np.linalg.norm(dataset["quaternion"][index])
        d.qpos[self.qids] = dataset["q"][index]
        d.qvel[self.dids] = dataset["dq"][index]
        # A free joint has world-frame translation and local-frame angular velocity.
        d.qvel[3:6] = dataset["gyro"][index]
        mj.mj_forward(m, d)
        contact = dataset["contact"][index].astype(bool)
        fallback = not contact.any()
        if fallback:
            contact[np.argmax(dataset["ff"][index])] = True
        heights = m.geom_size[self.feet, 0] - d.geom_xpos[self.feet, 2]
        # Lowest foot touches the plane; prevents initial interpenetration.
        height = float(np.max(heights) + self.height_offset)
        d.qpos[2] = height
        mj.mj_forward(m, d)
        velocities = []
        jac = np.zeros((3, m.nv))
        jacrot = np.zeros_like(jac)
        for foot in self.feet[contact]:
            mj.mj_jacGeom(m, d, jac, jacrot, foot)
            velocities.append(jac @ d.qvel)
        d.qvel[:3] = -np.mean(velocities, axis=0)
        initial_velocity = d.qvel[:3].copy()
        mj.mj_forward(m, d)
        return {"height_m": height, "velocity_world_mps": initial_velocity.tolist(),
                "contact_count": int(contact.sum()), "contact_fallback": fallback}

    def forces(self):
        forces = np.zeros((4, 3))
        wrench = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if self.floor not in (c.geom1, c.geom2):
                continue
            for j, foot in enumerate(self.feet):
                if foot in (c.geom1, c.geom2):
                    self.mj.mj_contactForce(self.model, self.data, i, wrench)
                    # Contact-frame force acts on geom2; convert and sum in WORLD.
                    sign = 1 if c.geom2 == foot else -1
                    forces[j] += sign * (c.frame.reshape(3, 3).T @ wrench[:3])
        return forces

    def rollout(self, dataset, clip_id, parameters=None, include_forces=True):
        p = DEFAULT_PARAMETERS if parameters is None else parameters
        start, end = dataset["clip_bounds"][clip_id]
        times = dataset["t"][start:end]
        sample_dt = float(times[1] - times[0])
        substeps = int(round(sample_dt / self.timestep))
        if substeps < 1 or not np.isclose(substeps * self.timestep, sample_dt, atol=1e-8):
            raise ValueError("Prepared dt must be an integer multiple of physics timestep")
        gain = np.tile([p["gain_" + j] for j in ("hip", "thigh", "calf")], 4)
        damping = np.tile([p["damping_" + j] for j in ("hip", "thigh", "calf")], 4)
        initialization = self.initialize(dataset, start, p)
        result = {k: [] for k in ("q", "dq", "gyro", "quaternion", "force_world")}
        total_steps = (len(times) - 1) * substeps
        query = times[0] + np.arange(total_steps + 1) * self.timestep
        commands = command_at(dataset["command_t"], query, p["delay_s"])
        d = self.data
        for step, ci in enumerate(commands):
            torque = (gain * dataset["command_kp"][ci] * (dataset["command_q"][ci] - d.qpos[self.qids])
                      + damping * dataset["command_kd"][ci] * (dataset["command_dq"][ci] - d.qvel[self.dids])
                      + dataset["command_tau"][ci])
            d.ctrl[self.aids] = np.clip(torque, -self.limits, self.limits)
            if step % substeps == 0:
                self.mj.mj_forward(self.model, d)
                result["q"].append(d.qpos[self.qids].copy())
                result["dq"].append(d.qvel[self.dids].copy())
                result["gyro"].append(d.qvel[3:6].copy())
                result["quaternion"].append(d.qpos[3:7].copy())
                if include_forces:
                    result["force_world"].append(self.forces())
            if step < total_steps:
                self.mj.mj_step(self.model, d)
                if not np.isfinite(d.qpos).all() or np.max(np.abs(d.qvel)) > 1e3 or d.warning.number.sum():
                    raise FloatingPointError(f"Unstable simulation at clip {clip_id}, step {step}")
        return {k: np.asarray(v) for k, v in result.items()}, initialization

    def metadata(self):
        return {"mujoco_version": self.mj.__version__, "physics_dt_s": self.timestep,
                "integrator": "implicitfast", "model_xml_sha256": hashlib.sha256(self.xml.encode()).hexdigest(),
                "nominal_base_mass_kg": self.base_mass, "height_offset_m": self.height_offset,
                "joint_armature": np.tile([.01, .01, .02], 4).tolist(),
                "torque_limit_nm": self.limits.tolist(), "foot_friction": .6,
                "foot_solimp": [.9, .95, .023, .5, 2],
                "source": str(self.model_path), "loss_scales": LOSS_SCALES}


def errors(prediction, dataset, clip_id):
    start, end = dataset["clip_bounds"][clip_id]
    # Initial state is copied by design: exclude it from every error metric.
    truth = slice(start + 1, end)
    error = {k: prediction[k][1:] - dataset[k][truth] for k in ("q", "dq", "gyro")}
    error["orientation"] = (rotation(dataset["quaternion"][truth]).inv()
                            * rotation(prediction["quaternion"][1:])).as_rotvec()
    return error


def prediction_cost(error):
    return float(sum(np.mean(value ** 2) / LOSS_SCALES[key] ** 2 for key, value in error.items()))
