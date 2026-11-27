"""Terrain scan sensor used by the rough-terrain environment."""

from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg


def make_terrain_scan_sensor() -> RayCastSensorCfg:
  return RayCastSensorCfg(
    name="terrain_scan",  # observations.py에서 이 이름으로 지형 높이를 읽는다.
    frame=ObjRef(type="body", name="", entity="robot"),  # go2_env.py에서 base_link로 연결.
    ray_alignment="yaw",  # 몸체의 수평 방향만 따라간다. roll/pitch 기울기는 반영하지 않는다.
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),  # 스캔 영역 크기/격자 간격 [m].
    max_distance=5.0,  # 레이가 접촉점을 찾는 최대 거리 [m].
    exclude_parent_body=True,  # 센서를 부착한 몸체 자체와의 충돌은 제외한다.
    debug_vis=True,  # 스캔 디버그 표시 활성화.
    viz=RayCastSensorCfg.VizCfg(show_normals=True),  # 맞은 표면의 법선도 표시한다.
  )
