"""Shared output locations for the model pipeline, analysis and experiments."""
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parent
ROOT = MODEL_ROOT.parent
FEATURE_SELECTION_RESULTS = MODEL_ROOT / 'results/feature_selection'
FEATURE_SELECTION_MODELS = MODEL_ROOT / 'models/feature_selection'
FEATURE_DIR = FEATURE_SELECTION_RESULTS / 'features'
ANALYSIS_RESULTS = MODEL_ROOT / 'results/analysis'


def resolve_recorded_path(path):
    """Locate a moved artifact while retaining its original manifest and hash."""
    path = Path(path)
    if path.is_absolute():
        try:
            relative = path.relative_to(ROOT)
        except ValueError:
            return path
    else:
        relative = path
    if not relative.parts:
        return path
    if relative.parts[0] == 'contact_force_velocity':
        return MODEL_ROOT.joinpath(*relative.parts[1:])
    if relative.parts[0] == 'Noise_prediction_model':
        rest = relative.parts[1:]
        if rest and rest[0] == 'models':
            return FEATURE_SELECTION_MODELS.joinpath(*rest[1:])
        if rest and rest[0] == 'model_evaluation':
            return FEATURE_SELECTION_RESULTS.joinpath(*rest[1:])
        if rest and rest[0] == 'Feature_extraction':
            if len(rest) > 1 and rest[1] == 'landing_dynamics':
                return ANALYSIS_RESULTS.joinpath(*rest[1:])
            if len(rest) > 1 and rest[1].startswith(('contact_event_', 'contact_force_impact_velocity')):
                return (ANALYSIS_RESULTS / 'contact_events').joinpath(*rest[1:])
            return FEATURE_DIR.joinpath(*rest[1:])
    return path
