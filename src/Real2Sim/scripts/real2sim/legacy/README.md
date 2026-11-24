# 과거 정책 재구성·물리 재생 실험

이 폴더의 코드는 현재 분석·UI 실행 경로에서 호출하지 않는다.
`real2sim_converter.py`는 과거 로그 형식의 정책 관측/action을 재구성하고,
`analyze_converter.py`와 `full_record_replay.py`는 MuJoCo 물리 재생을 수행한다.
현재 데이터 처리는 상위 `pipeline.py`, UI는 `replay/serve.py`를 사용한다.
