"""Existing time-series CSV schema and plotting, independent of synchronization."""
from __future__ import annotations
import csv
import os
from pathlib import Path
import numpy as np
from common_time import TrialReader

KNEE_NAMES = ("fr", "fl", "rr", "rl")

MOTOR_NAMES = tuple(f"{foot}_{joint}" for foot in KNEE_NAMES for joint in ("hip", "thigh", "calf"))

VECTOR_NAMES = {
    "ls_ff": ("fr", "fl", "rr", "rl"),
    "ls_acc": ("x", "y", "z"),
    "ls_gyro": ("x", "y", "z"),
    "ls_rpy": ("roll", "pitch", "yaw"),
    "ls_quaternion": ("w", "x", "y", "z"),
    "ls_tick": ("value",),
    "ls_q": MOTOR_NAMES,
    "ls_dq": MOTOR_NAMES,
    "ls_tau": MOTOR_NAMES,
    "sp_vel": ("x", "y", "z"),
    "sp_foot_vel": tuple(
        f"{foot}_{axis}" for foot in KNEE_NAMES for axis in ("x", "y", "z")
    ),
    "knee_q": KNEE_NAMES,
    "knee_dq": KNEE_NAMES,
    "knee_tau": KNEE_NAMES,
}

PLOT_GROUPS = (
    ("Foot force", tuple(f"ls_ff_{name}" for name in ("fr", "fl", "rr", "rl"))),
    ("Acceleration", ("ls_acc_x", "ls_acc_y", "ls_acc_z")),
    ("Gyroscope", ("ls_gyro_x", "ls_gyro_y", "ls_gyro_z")),
    ("RPY", ("ls_rpy_roll", "ls_rpy_pitch", "ls_rpy_yaw")),
    ("Quaternion", tuple(f"ls_quaternion_{name}" for name in ("w", "x", "y", "z"))),
    ("Sport velocity", ("sp_vel_x", "sp_vel_y", "sp_vel_z")),
    ("Foot-tip velocity", tuple(
        f"sp_foot_vel_{foot}_{axis}"
        for foot in KNEE_NAMES for axis in ("x", "y", "z")
    )),
    ("All motor positions", tuple(f"ls_q_{name}" for name in MOTOR_NAMES)),
    ("All motor velocities", tuple(f"ls_dq_{name}" for name in MOTOR_NAMES)),
    ("All motor estimated torques", tuple(f"ls_tau_{name}" for name in MOTOR_NAMES)),
)

def csv_columns() -> list[str]:
    columns = [
        "trial", "command_vx", "command_vy", "command_vyaw", "time_abs", "time_from_move_start",
        "audio_rms_20ms", "audio_dbfs_20ms", "audio_peak_2ms",
    ]
    for key, names in VECTOR_NAMES.items():
        columns.extend(f"{key}_{name}" for name in names)
    columns.extend(("sp_vel_x_smooth", "sp_acc_x", "audio_delay_s", "is_steady", "segment"))
    return columns

def iter_rows(
    trial: TrialReader,
    data: dict[str, np.ndarray],
    state: dict[str, np.ndarray],
    selection: str = "walking",
):
    delay = float(data["audio_delay_s"])
    from common_time import recorded_commands
    commands = recorded_commands(trial, data['time_abs'])
    mask = state["steady"] if selection == "steady" else data["valid"] & (state["move_mask"] if selection == "walking" else True)
    segments = np.cumsum(np.r_[False, np.diff(data["time_abs"][mask]) > 1.5/float(data["target_hz"])])
    for segment, i in zip(segments, np.flatnonzero(mask)):
        row: list[object] = [
            trial.name, *commands[i].tolist(), data["time_abs"][i], data["time"][i],
            data["audio_rms_20ms"][i], data["audio_dbfs_20ms"][i],
            data["audio_peak_2ms"][i],
        ]
        for key in VECTOR_NAMES:
            row.extend(np.atleast_1d(data[key][i]).tolist())
        row.extend((state["sp_vel_x_smooth"][i], state["sp_acc_x"][i], delay, int(state["steady"][i]), int(segment)))
        yield row

def write_csv(path: Path, columns: list[str], rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temporary = path.with_name(path.name+f".tmp.{os.getpid()}")
    try:
        with temporary.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            for row in rows:
                writer.writerow(row)
                count += 1
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(path)
    return count

def _plot_feature_values(values, trial_name, hz, figure_dir=None, save=True, show=False):
    """Render selected rows and break lines at missing time samples."""
    import matplotlib.pyplot as plt
    time=np.asarray(values['time_from_move_start'],dtype=float)
    if not len(time):
        print(f'{trial_name}: no selected rows to plot');return
    breaks=np.r_[False,np.diff(time)>1.5/hz]
    plot_time=time.copy();plot_time[breaks]=np.nan
    audio=np.asarray(values['audio_dbfs_20ms'],dtype=float).copy();audio[breaks]=np.nan
    fig,axes=plt.subplots(len(PLOT_GROUPS),1,figsize=(15,18),sharex=True)
    for index,(title,names) in enumerate(PLOT_GROUPS):
        ax=axes[index]
        for name in names:
            feature=np.asarray(values[name],dtype=float).copy();feature[breaks]=np.nan
            ax.plot(plot_time,feature,lw=.65,label=name)
        ax.set_ylabel(title);ax.grid(alpha=.2);ax.legend(loc='upper left',ncol=min(4,len(names)),fontsize=7)
        noise_ax=ax.twinx();noise_ax.plot(plot_time,audio,color='black',alpha=.28,lw=.7,label='audio dBFS')
        noise_ax.set_ylabel('dBFS')
        if index==0:noise_ax.legend(loc='upper right',fontsize=7)
    axes[-1].set_xlabel('Time from move_start (s)')
    fig.suptitle(f'{trial_name}: preprocessed robot states and audio, selected intervals')
    fig.tight_layout(rect=(0,0,1,.985))
    if save:
        folder=Path(figure_dir) if figure_dir is not None else Path("preprocessed/1104/figure")/trial_name
        folder.mkdir(parents=True,exist_ok=True)
        fig.savefig(folder/'timeseries.png',dpi=140)
        print(f'plot: {folder / "timeseries.png"}',flush=True)
    if show:plt.show()
    plt.close(fig)

def plot_processed_data(trial,data,state,selection,hz,figure_dir,show=False):
    """Same selected rows as CSV generation, without a large CSV round trip."""
    mask=state['steady'] if selection=='steady' else data['valid'] & (state['move_mask'] if selection=='walking' else True)
    values=dict(time_from_move_start=data['time'][mask],audio_dbfs_20ms=data['audio_dbfs_20ms'][mask])
    required={name for _,names in PLOT_GROUPS for name in names}
    for key,names in VECTOR_NAMES.items():
        if not any(f'{key}_{name}' in required for name in names):continue
        array=np.asarray(data[key]).reshape(len(data['time']),-1)
        for index,name in enumerate(names):values[f'{key}_{name}']=array[mask,index]
    _plot_feature_values(values,trial.name,hz,figure_dir,show=show)
