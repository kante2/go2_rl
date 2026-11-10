"""Readable synchronization figures. Rendering never changes alignment or data."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

STYLE = {'font.family': 'DejaVu Sans', 'axes.unicode_minus': False,
         'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 11,
         'axes.spines.top': False, 'axes.spines.right': False,
         'figure.facecolor': 'white', 'axes.facecolor': 'white',
         'text.color': '#253449', 'axes.labelcolor': '#526174'}
GREEN, AMBER, RED, BLUE = '#218568', '#b77916', '#be4d55', '#3178b5'
LEG_INDEX = {'fr': 0, 'fl': 1, 'rr': 2, 'rl': 3}
OBSOLETE = ('clock_quality.png', 'timeseries.png', 'contact_force_velocity.png')


def assessment(summary, middle=None):
    if not summary['passed']:
        return 'Review needed', RED
    if middle is not None and not all(row['passed'] for row in middle):
        return 'Middle review', AMBER
    return ('Good' if middle else 'Start/end passed'), GREEN


def _number(value, absolute=False):
    if value is None or not np.isfinite(value):
        return 'Unavailable'
    return f'{abs(value) if absolute else value:.2f} ms'


def _selected(events):
    start = [e for e in events if e['window'] == 'start']
    end = [e for e in events if e['window'] == 'end']
    return (start[0] if start else None, end[-1] if end else None)


def _sound(ax, item, audio, delay, tolerance, detail=False):
    if item is None:
        ax.text(.5, .5, 'No contact detected', ha='center', transform=ax.transAxes)
        return
    center = item['sensor_time']
    raw_x = (audio['time']-center)*1000
    corrected_x = raw_x-delay*1000
    mask = abs(corrected_x) <= 110
    ax.axvspan(-tolerance, tolerance, color='#edf0f3', zorder=0)
    ax.axvline(0, color='#253449', lw=1.5)
    if detail:
        original = abs(raw_x) <= 110
        ax.plot(raw_x[original], audio['envelope'][original], color='#c08a57',
                lw=1.1, alpha=.75, label='Original audio')
    ax.plot(corrected_x[mask], audio['envelope'][mask], color=BLUE, lw=1.8,
            label='Aligned audio')
    if item.get('matched') and item.get('audio_time') is not None:
        residual = (item['audio_time']-delay-center)*1000
        ax.axvline(residual, color=BLUE, ls='--', lw=1.2)
        text = f"Sound onset: {abs(residual):.1f} ms {'early' if residual < 0 else 'late'}"
    else:
        text = 'No reliable sound match for this contact'
    ax.text(.98, .94, text, ha='right', va='top', transform=ax.transAxes,
            fontsize=11, bbox=dict(facecolor='white', edgecolor='none', alpha=.9))
    ax.set(xlim=(-100, 100), xlabel='Time relative to sensor contact (ms)', ylabel='Audio amplitude (RMS)')
    ax.set_xticks([-100, -50, 0, 50, 100]); ax.grid(axis='y', alpha=.15)
    ax.set_ylim(bottom=0)
    if detail:
        ax.legend(loc='upper left', fontsize=9)


def plot_trial_figures(trial, summary, events, audio, output, middle=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    selected = _selected(events)
    tolerance = summary['config']['tolerance_s']*1000
    delay = summary['applied_audio_delay_s']
    status, color = assessment(summary, middle)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(11, 9.2))
        grid = fig.add_gridspec(2, 1, left=.10, right=.95, bottom=.16, top=.67, hspace=.72)
        fig.text(.10, .945, f"{summary['trial']}  |  {summary.get('split', '')}", fontsize=22, weight='bold')
        fig.text(.95, .945, status, ha='right', fontsize=16, color=color,
                 bbox=dict(boxstyle='round,pad=.5', facecolor=color+'14', edgecolor='none'))
        direction = 'Later by' if delay < 0 else 'Earlier by'
        shift = f'{direction} {abs(delay)*1000:.2f} ms' if summary['correction_accepted'] else 'Not applied'
        match = f"Start {summary['calibration']['inliers']}/{summary['calibration']['events']}  ·  End {summary['validation']['inliers']}/{summary['validation']['events']}"
        cards = [('Audio shift', shift), ('Median end error', _number(summary['end_residual_ms'], True)), ('Contact matches', match)]
        for x, (label, value) in zip((.10, .39, .68), cards):
            fig.text(x, .855, label, color='#718096', fontsize=12)
            fig.text(x, .807, value, fontsize=17 if x < .68 else 12, weight='bold')
        evidence = f"End-match absolute error, P90: {_number(summary['end_abs_error_p90_ms'], True)}"
        evidence += f"    |    Middle check: {sum(r['passed'] for r in middle)}/{len(middle)} windows passed" if middle else '    |    Middle windows checked separately'
        fig.text(.10, .74, evidence, fontsize=11, color='#526174')
        for i, (title, item) in enumerate(zip(('First contact', 'Last contact'), selected)):
            ax = fig.add_subplot(grid[i])
            _sound(ax, item, audio, delay, tolerance)
            legs = ' + '.join(leg.upper() for leg in item['legs']) if item else ''
            ax.set_title(f'{title}   {legs}', loc='left', pad=12)
        fig.text(.10, .045, f'Black = sensor contact  |  Blue = aligned audio  |  Shading = ±{tolerance:g} ms\nAssessment uses multiple nearby contacts. The plots show the actual first and last contacts.',
                 fontsize=10, color='#718096', linespacing=1.7)
        fig.savefig(output/'synchronization.png', dpi=160); plt.close(fig)

        fig, axes = plt.subplots(3, 2, figsize=(13, 10), gridspec_kw={'height_ratios': [1, 1.4, .65]})
        time, force = trial.load('ls_t'), trial.load('ls_ff')
        for column, (title, item) in enumerate(zip(('First contact', 'Last contact'), selected)):
            axes[0, column].set_title(title, loc='left', pad=12)
            if item is None:
                for ax in axes[:, column]:
                    ax.text(.5, .5, 'No contact', ha='center', transform=ax.transAxes)
                continue
            center = item['sensor_time']; mask = abs(time-center) <= .1
            for leg in dict.fromkeys(item['legs']):
                line, = axes[0, column].plot((time[mask]-center)*1000, force[mask, LEG_INDEX[leg]], lw=1.4, label=leg.upper())
                threshold = summary['contact_thresholds'][leg]['on']
                axes[0, column].axhline(threshold, color=line.get_color(), ls=':', alpha=.6)
            axes[0, column].axvline(0, color='#253449', lw=1.2)
            axes[0, column].set(xlim=(-100, 100), ylabel='Foot force (sensor units)')
            axes[0, column].legend(fontsize=9, ncol=2)
            _sound(axes[1, column], item, audio, delay, tolerance, detail=True)
            ax = axes[2, column]
            for candidate in item['candidates']:
                x = (candidate['audio_time']-delay-center)*1000
                ax.scatter(x, candidate['prominence'], s=30, color='#b9c2ce')
            if item.get('matched'):
                x = (item['audio_time']-delay-center)*1000
                ax.scatter(x, item['prominence'], s=75, color=BLUE, zorder=3)
            ax.axvline(0, color='#253449'); ax.axvspan(-tolerance, tolerance, color='#edf0f3', zorder=0)
            ax.set(xlim=(-100, 100), ylim=(0, None), xlabel='Time relative to sensor contact (ms)', ylabel='Candidate prominence')
            ax.text(.98, .9, 'Blue: selected  /  Gray: other candidates', transform=ax.transAxes, ha='right', fontsize=9)
        fig.suptitle(f"{summary['trial']}  ·  Detailed diagnostics", fontsize=20, x=.08, ha='left')
        fig.subplots_adjust(left=.08, right=.97, top=.91, bottom=.10, hspace=.56, wspace=.25)
        fig.text(.08, .025, 'Dotted horizontal lines = contact thresholds. Force, audio and candidates share the same contact reference.\nReceive-time diagnostics remain in synchronization.json.', fontsize=10, color='#718096')
        fig.savefig(output/'synchronization_detail.png', dpi=160); plt.close(fig)
    for name in OBSOLETE:
        (output/name).unlink(missing_ok=True)


def plot_overview(summaries, output, middle_by_trial=None):
    middle_by_trial = middle_by_trial or {}
    statuses = [assessment(s, middle_by_trial.get(s['trial'])) for s in summaries]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(12, max(9, len(summaries)*.32+3)))
        tolerance = summaries[0]['config']['tolerance_s']*1000
        ax.axvspan(-tolerance, tolerance, color='#edf0f3')
        ax.axvline(0, color='#718096', lw=1)
        for y, (s, (status, color)) in enumerate(zip(summaries, statuses)):
            error, p90 = s['end_residual_ms'], s['end_abs_error_p90_ms']
            if p90 is not None:
                ax.plot([-p90, p90], [y, y], color=color, lw=4, alpha=.3, solid_capstyle='round')
            if error is not None:
                ax.scatter(error, y, color=color, s=45, marker='o' if s['passed'] else 'x', zorder=3)
            ax.text(1.02, y, status, va='center', color=color, fontsize=10,
                    transform=ax.get_yaxis_transform())
        ax.set_yticks(range(len(summaries)), [f"{s['split']}  ·  {s['trial']}" for s in summaries])
        maximum = max([abs(s['end_residual_ms'] or 0) for s in summaries]+[s['end_abs_error_p90_ms'] or 0 for s in summaries]+[12])+3
        ax.set(xlim=(-maximum, maximum), ylim=(len(summaries)-.4, -.6), xlabel='End residual after alignment (ms)')
        ax.grid(axis='x', alpha=.15); ax.tick_params(axis='y', length=0, pad=12)
        fig.text(.16, .96, 'Synchronization across sequences', fontsize=22, weight='bold')
        counts = {key: sum(status == key for status, _ in statuses) for key in ('Good', 'Start/end passed', 'Middle review', 'Review needed')}
        fig.text(.16, .92, '   ·   '.join(f'{key}: {count}' for key, count in counts.items() if count), fontsize=13)
        fig.subplots_adjust(left=.16, right=.79, top=.87, bottom=.12)
        fig.text(.16, .045, f'Dot = median end residual  |  Bar = P90 absolute end error  |  Shading = ±{tolerance:g} ms\nGood includes the middle checks. Too few matches require review even when the residual is small.', fontsize=10, color='#718096', linespacing=1.7)
        fig.savefig(Path(output)/'synchronization_overview.png', dpi=160); plt.close(fig)


def load_middle_checks(figure_dir):
    import pandas as pd
    path = Path(figure_dir)/'middle_synchronization_check.csv'
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    return {name: group.to_dict('records') for name, group in frame.groupby('trial')}
