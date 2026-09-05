# FacilityPath v1.1 — 시설물 드론 자동비행경로 생성

㈜드론아이디 AI연구소 · DID-DP-2026-0829 부속 · 2026-09-05

## 1. 처리 순서와 모듈 대응 (일관된 6단계)

모든 실행 모드가 아래 순서를 그대로 따릅니다. 파일명이 단계와 1:1로 대응하도록 재정리했습니다.

| 단계 | 모듈 | 역할 |
|---|---|---|
| **[1] 기준면 입력** | `io_model.py` | OBJ/PLY/STL 메시 **또는** DSM(.tif/.asc/.xyz) → 로컬 ENU 메시 + 지오레퍼런스 원점 |
| **[1'] 기존 경로 입력** | `io_flightpath.py` | UgCS KML / DJI WPML(.kmz) / CSV → 로컬 ENU 웨이포인트 (adapt 모드 전용) |
| **[2] 대상면 분석** | `surface_analyzer.py` | 경사각 범위로 촬영면 추출, 경사각 α·주향·범위·정점 인덱스 |
| **[3] 촬영 파라미터** | `camera.py` | GSD ↔ 접근거리, 트리거/라인 간격 |
| **[4] 경로 생성** | `planner_base.py` + `planner_slope.py` | generate: 등고선형 서펜타인 신규 설계 |
| | `path_adapter.py` | adapt: 기존 경로를 기준면 최근접점 + 법선 오프셋으로 재배치 |
| | `tilt_solver.py` | 공용: 짐벌 피치 φ = α − 90°, 헤딩 = 면 방향 |
| **[5] 확정** | `planner_base.finalize()` | 표면 최소이격 검사 → 소티 분할(라인 경계) → 통계 |
| **[6] 출력** | `kml_writer.py` | Earth 검토 KML + 소티별 DJI WPML(.kmz) + CSV + 요약 JSON |
| 진입점 | `facilitypath.py` | 위 6단계를 순서대로 호출하는 CLI (generate / adapt) |

## 2. 두 가지 입력 방식

### A. 3D 모델·DSM → 경로 신규 생성 (`generate`)

```powershell
# iTwin Capture Modeler OBJ (투영좌표 EPSG:5186)
python facilitypath.py generate slope.obj --epsg 5186 --gsd 0.5 --out run1

# DSM (ESRI ASCII / GeoTIFF*) — 사면 앞뒤 평지는 경사각 필터로 자동 제외
python facilitypath.py generate site_dsm.asc --epsg 5186 --slope-min 30 --dsm-step 2 --out run2

# 로컬 좌표 OBJ는 기준점(lat,lon,h) 지정
python facilitypath.py generate local.obj --anchor 36.2243694,127.2765611,121 --out run3
```
\* GeoTIFF는 `pip install rasterio` 필요(선택). ASC/XYZ는 추가 설치 없음.

### B. 기존 비행경로 → 시설물 형상 적응 (`adapt`)

UgCS 평면 그리드(연직 촬영)나 수직 파사드 경로를 기준면에 맞춰 **법선 정렬 경로로 재배치**합니다. 웨이포인트 순서와 라인 구조는 보존됩니다.

```powershell
# UgCS 그리드 KML(절대고도) → 사면 법선 정렬, 접근거리는 GSD로 산출
python facilitypath.py adapt ugcs_grid.kml --model slope.obj --epsg 5186 --gsd 0.5 --alt-mode absolute --out run4

# 파사드 경로 — 원래 이격거리 유지, 법선(짐벌·헤딩)만 정렬
python facilitypath.py adapt facade.kmz --model bridge.obj --epsg 5186 --keep-distance --out run5

# ENU 좌표 CSV (헤더 E,N,U) 직접 입력
python facilitypath.py adapt path_enu.csv --model slope.obj --epsg 5186 --approach 20 --out run6
```

적응 알고리즘: 각 웨이포인트에 대해 ① 기준면 최근접점·국부 법선 → ② 새 위치 = 면상점 + 법선 × 접근거리 → ③ φ = α − 90° → ④ 기준면에서 `--max-snap`(기본 80 m) 초과 이격 점 제외 → ⑤ 재배치 후 과밀 점 병합.

## 3. 출력물 (`--out` 폴더)

`mission_review.kml`(Earth), `mission_sortie{N}.kmz`(Pilot 2), `mission_waypoints.csv`, `mission_summary.json`(경사각·접근거리·소티·경고·왕복오차).

## 4. 검증 (환경 변경 시 반드시 재실행)

```powershell
python verify_prototype.py        # M2  10/10
python verify_planner_slope.py    # M3   7/7
python verify_kml_writer.py       # M4   7/7
python verify_io_adapter.py       # v1.1 입력 2종 13/13 (CLI 통합)
```

## 5. 현재 한계 (다음 단계)

- `--facility bridge / dam` 전략은 미구현 — slope 전략으로 대체 실행되며 경고 출력.
- DJI 열거값(기체 60 / 페이로드 50)은 Pilot 2 내보내기 샘플과 대조 필요.
- 3MX/3SM은 미지원 — iTwin에서 OBJ 재출력.
