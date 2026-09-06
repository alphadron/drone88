# CLAUDE.md — FacilityPath 프로젝트 작업 지침

이 파일은 Claude Code가 이 저장소에서 작업할 때 자동으로 읽는 프로젝트 규칙이다.
㈜드론아이디 AI연구소 · DID-DP-2026-0829

## 프로젝트 개요
시설물(급경사지·교량·댐) 안전점검용 드론 자동비행경로 생성기.
3D 모델/DSM 또는 기존 경로를 입력받아 경사면 법선 정렬 경로를 만들고
Google Earth KML + DJI Pilot 2 WPML(.kmz)로 출력한다.

핵심 공식: 짐벌 피치 φ = α − 90° (α: 경사각, 평지 0° / 수직벽 90°).
좌표 규약: 내부 계산은 로컬 ENU(m). 출력 시 AEQD 지역투영으로 WGS84 복원.

## 처리 순서 (모듈 = 단계, 이 순서를 깨지 말 것)
[1] io_model / io_flightpath → [2] surface_analyzer → [3] camera
→ [4] planner_slope 또는 path_adapter (+ tilt_solver) → [5] planner_base.finalize
→ [6] kml_writer.  진입점은 facilitypath.py 하나.

## 모델 사용 정책 (비용 효율 — 반드시 준수)
작업 성격에 따라 모델을 선택한다. 기본은 **Sonnet**이며, 시작 시 /model 로 확인한다.
| 작업 | 모델 |
|---|---|
| 새 전략 클래스 설계(planner_bridge/dam), 알고리즘·수치 안정성 판단, 좌표계 문제, 특허 관련 논리 | Opus 계열 (상위) |
| 기능 구현, 리팩터링, 테스트 작성, 버그 수정, 문서화 | Sonnet (기본) |
| 포맷 정리, 이름 변경, 주석·docstring 보강, 반복 보일러플레이트 | Haiku |
서브에이전트(.claude/agents/)를 만들 때도 같은 기준으로 model 필드를 지정한다.
상위 모델이 필요하다고 판단되면 이유를 한 줄 말하고 전환을 제안한다.

## 환경
- Python 3.14.7 / Windows 11 / 가상환경 `.venv` (활성화: `.\.venv\Scripts\Activate.ps1`)
- 의존성: requirements-py314.txt (vtk는 반드시 >=9.6,<9.7)
- 코어 개발 시 3D뷰(pyvista)·UI(streamlit)는 불필요. import 하지 말 것.

## 필수 규칙
1. **코드 수정 후 반드시 검증 4종을 실행하고 결과를 보고한다.**
   `python verify_prototype.py` (10/10) · `verify_planner_slope.py` (7/7)
   · `verify_kml_writer.py` (7/7) · `verify_io_adapter.py` (13/13)
   하나라도 FAIL이면 커밋하지 않는다.
2. 새 기능에는 verify_*.py 형식의 자동 검증(PASS/FAIL 카운트 출력)을 함께 추가한다.
3. 의존성 추가는 requirements-py314.txt에 사유 주석과 함께 기록한다. rtree, open3d 등
   네이티브 의존성은 추가 전에 사용자에게 묻는다.
4. 파일 인코딩: .py는 UTF-8, .ps1은 UTF-8 with BOM 또는 ASCII 전용.
5. 커밋 메시지는 한국어, 형식 `[M단계] 요약` (예: `[M3] planner_bridge 파사드 전략 추가`).
6. 산출물(output/, verify_io/, *.kmz, *.png)은 커밋하지 않는다 (.gitignore 참조).
7. DJI WPML 열거값(기체 60/페이로드 50)은 미검증 상수 — 변경 시 근거를 남긴다.

## 하지 말 것
- verify 결과를 실행하지 않고 "통과할 것"이라고 추정 보고
- 모듈 처리 순서를 바꾸거나 facilitypath.py 외에 새 진입점 추가
- 한글 주석 삭제, 모듈 헤더의 문서번호 제거
