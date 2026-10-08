"""Terrain scan sensor used by the rough-terrain environment."""

from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg


def make_terrain_scan_sensor() -> RayCastSensorCfg:
  return RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="", entity="robot"),  # Set per-robot.
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=5.0,
    exclude_parent_body=True,
    debug_vis=True,
    viz=RayCastSensorCfg.VizCfg(show_normals=True),
  )
