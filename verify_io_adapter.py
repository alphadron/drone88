#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 verify_io_adapter.py — FacilityPath v1.1 입력 2종 통합 검증
--------------------------------------------------------------------------------
 (주)드론아이디 AI연구소 · DID-DP-2026-0829 부속 (2026-09-05)
--------------------------------------------------------------------------------
 시나리오 (전부 합성 데이터, 60° 남향 사면, EPSG:5186 투영좌표):
   [D] DSM 경로 : 사면 DSM(.asc) 생성 → generate 모드 → α·φ·KML 검증
   [G] 그리드 적응 : UgCS식 평면 연직 그리드 KML 생성(고도 일정, 짐벌 −90°)
                     → adapt 모드 → 모든 점이 면에서 d 이격, φ=−30°, 헤딩 0°
   [F] 파사드 적응 : 수직 파사드 경로 CSV(ENU) 생성 → adapt --keep-distance
                     → 원래 이격 유지, 법선 정렬만 수행
   [O] OBJ 경로  : 투영좌표 OBJ 저장 → generate → 기존 검증과 동일 결과
 판정 항목 12건. 산출물: verify_io/ 폴더
================================================================================
"""

import json
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import simplekml
import trimesh
from pyproj import CRS, Transformer

from surface_analyzer import make_synthetic_slope
from kml_writer import GeoAnchor, EnuConverter

OUT = "verify_io"
ALPHA, ASPECT = 60.0, 180.0
EPSG = 5186
# 현장 기준점(투영좌표계 5186)에 사면을 놓는다
X0, Y0, Z0 = 214_000.0, 425_000.0, 121.0

checks = []
def chk(name, ok, detail):
    checks.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<30s} {detail}")


def run_cli(*args):
    cmd = [sys.executable, "facilitypath.py", *args]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-1500:])
    return r.returncode


def load_summary(folder):
    return json.load(open(os.path.join(folder, "mission_summary.json"), encoding="utf-8"))


# ── 공통: 투영좌표 사면 메시 ──
mesh_local = make_synthetic_slope(ALPHA, ASPECT, width_m=120, length_m=40, grid=40)
mesh_proj = mesh_local.copy()
mesh_proj.vertices = mesh_local.vertices + np.array([X0, Y0, Z0])
os.makedirs(OUT, exist_ok=True)

print("=" * 88)
print("  FacilityPath v1.1 검증 — 입력 2종 (DSM / 기존경로 적응) + 통합 파이프라인")
print("=" * 88)

# ============================ [O] OBJ → generate ==============================
print("\n[O] 투영좌표 OBJ → generate")
obj = os.path.join(OUT, "slope_5186.obj"); mesh_proj.export(obj)
rc = run_cli("generate", obj, "--epsg", str(EPSG), "--gsd", "0.5",
             "--slope-min", "10", "--out", os.path.join(OUT, "o"))
so = load_summary(os.path.join(OUT, "o")) if rc == 0 else {}
chk("O1 OBJ 파이프라인 완주", rc == 0, f"rc={rc}")
chk("O2 대상면 경사각", rc == 0 and abs(so["surface"]["slope_deg"] - ALPHA) < 0.5,
    f"α={so.get('surface',{}).get('slope_deg')}° (기대 {ALPHA})")
chk("O3 짐벌 φ=α−90°", rc == 0 and max(abs(p - (ALPHA-90)) for p in so["pitch_range_deg"]) < 0.5,
    f"φ 범위 {so.get('pitch_range_deg')} (기대 {ALPHA-90:.0f})")
tr = Transformer.from_crs(CRS.from_epsg(EPSG), CRS.from_epsg(4326), always_xy=True)
lon_e, lat_e = tr.transform(X0, Y0)
chk("O4 ENU 원점 지오레퍼런싱", rc == 0 and
    abs(so["anchor"]["lat"] - lat_e) < 1e-3 and abs(so["anchor"]["lon"] - lon_e) < 1e-3,
    f"anchor {so.get('anchor',{}).get('lat',0):.5f},{so.get('anchor',{}).get('lon',0):.5f}")

# ============================ [D] DSM(.asc) → generate ========================
print("\n[D] DSM(.asc) → generate")
# 사면을 포함하는 정규 격자 DSM: 사면 앞뒤는 평지(0/상단높이)로 채움
cs = 1.0
xs = np.arange(X0 - 80, X0 + 80, cs); ys = np.arange(Y0 - 60, Y0 + 60, cs)
X, Y = np.meshgrid(xs, ys)
# 남향 60° 사면: 북(+y)으로 갈수록 높아짐, y∈[0, L·cos α] 구간이 사면
L = 40.0; ymax = L * np.cos(np.radians(ALPHA)); zmax = L * np.sin(np.radians(ALPHA))
yy = Y - Y0
Z = np.where(yy < 0, 0.0, np.where(yy > ymax, zmax, yy * np.tan(np.radians(ALPHA)))) + Z0
asc = os.path.join(OUT, "site_dsm.asc")
with open(asc, "w") as f:
    f.write(f"ncols {len(xs)}\nnrows {len(ys)}\nxllcenter {xs[0]}\nyllcenter {ys[0]}\n"
            f"cellsize {cs}\nNODATA_value -9999\n")
    np.savetxt(f, Z[::-1], fmt="%.3f")               # 북→남
rc = run_cli("generate", asc, "--epsg", str(EPSG), "--gsd", "0.5",
             "--slope-min", "30", "--min-area", "100", "--out", os.path.join(OUT, "d"))
sd = load_summary(os.path.join(OUT, "d")) if rc == 0 else {}
chk("D1 DSM 파이프라인 완주", rc == 0, f"rc={rc}")
chk("D2 DSM 사면 경사각 추출", rc == 0 and abs(sd["surface"]["slope_deg"] - ALPHA) < 1.0,
    f"α={sd.get('surface',{}).get('slope_deg')}° (평지 자동 제외)")
chk("D3 DSM 짐벌 φ", rc == 0 and max(abs(p - (ALPHA-90)) for p in sd["pitch_range_deg"]) < 1.0,
    f"φ 범위 {sd.get('pitch_range_deg')}")
# DSM에는 사면 앞 평지가 있어 하단 라인의 지면 이격이 접근거리보다 작아지는 것이 물리적으로 정상
chk("D4 DSM 최소이격 ∈ [안전버퍼, 접근거리]", rc == 0 and
    5.0 <= sd["min_clearance_m"] <= sd["approach_dist_m"] + 0.5,
    f"{sd.get('min_clearance_m')} m (버퍼 5 ≤ · ≤ d {sd.get('approach_dist_m')})")

# ============================ [G] 평면 그리드 KML → adapt =====================
print("\n[G] UgCS식 평면 연직 그리드 KML → adapt")
anchor = GeoAnchor(lat=lat_e, lon=lon_e, h=Z0)
conv = EnuConverter(anchor)
# 사면 상공 고도 60m 일정, 동서 방향 왕복 6라인 (연직 촬영 가정)
gx = np.linspace(-50, 50, 21); gy = np.linspace(2, 18, 6)
grid = []
for i, y in enumerate(gy):
    row = [[x, y, 60.0] for x in (gx if i % 2 == 0 else gx[::-1])]
    grid += row
grid = np.array(grid)
llh = conv.to_wgs84(grid)
k = simplekml.Kml(); ls = k.newlinestring(coords=[tuple(p) for p in llh])
ls.altitudemode = simplekml.AltitudeMode.absolute
gkml = os.path.join(OUT, "ugcs_grid.kml"); k.save(gkml)
rc = run_cli("adapt", gkml, "--model", obj, "--epsg", str(EPSG), "--gsd", "0.5",
             "--alt-mode", "absolute", "--slope-min", "10", "--out", os.path.join(OUT, "g"))
sg = load_summary(os.path.join(OUT, "g")) if rc == 0 else {}
chk("G1 그리드 적응 완주", rc == 0, f"rc={rc}, WP {sg.get('n_waypoints')}")
chk("G2 적응 후 φ = −30° (연직→법선정렬)", rc == 0 and
    max(abs(p - (ALPHA-90)) for p in sg["pitch_range_deg"]) < 0.5,
    f"φ 범위 {sg.get('pitch_range_deg')}")
chk("G3 적응 후 면 이격 = 접근거리", rc == 0 and
    abs(sg["min_clearance_m"] - sg["approach_dist_m"]) < 1.0,
    f"{sg.get('min_clearance_m')} ≈ {sg.get('approach_dist_m')} m")
chk("G4 라인 구조 보존(6라인)", rc == 0 and sg["n_lines"] == 6, f"라인 {sg.get('n_lines')}")

# ============================ [F] 수직 파사드 CSV → adapt --keep-distance =====
print("\n[F] 수직 파사드 경로 CSV(ENU) → adapt --keep-distance")
# 사면 전방 15 m 지점에서 수평으로 촬영하던(피치 0°) 경로: 사면 중심 높이별 3라인
n_ax = mesh_local.face_normals.mean(axis=0); n_ax /= np.linalg.norm(n_ax)
lines = []
for h in (8.0, 18.0, 28.0):
    ps = mesh_local.vertices[np.abs(mesh_local.vertices[:, 2] - h) < 0.6]
    ps = ps[np.argsort(ps[:, 0])][::3]
    lines.append(ps + n_ax * 15.0)
fac = np.vstack(lines)
# io_model의 ENU 원점 규약(평균 x,y · 최저 z)에 맞춰 CSV 좌표 정렬
fac = fac - np.array([mesh_local.vertices[:, 0].mean(),
                      mesh_local.vertices[:, 1].mean(),
                      mesh_local.vertices[:, 2].min()])
fcsv = os.path.join(OUT, "facade_enu.csv")
np.savetxt(fcsv, fac, delimiter=",", header="E,N,U", comments="", fmt="%.3f")
rc = run_cli("adapt", fcsv, "--model", obj, "--epsg", str(EPSG), "--gsd", "0.5",
             "--keep-distance", "--slope-min", "10", "--out", os.path.join(OUT, "f"))
sf = load_summary(os.path.join(OUT, "f")) if rc == 0 else {}
chk("F1 파사드 적응(--keep-distance)", rc == 0 and abs(sf["min_clearance_m"] - 15.0) < 1.0,
    f"이격 {sf.get('min_clearance_m')} m (원래 15 m 유지)")

# ── 결과 ──
fails = sum(not ok for _, ok, _ in checks)
print("\n" + "=" * 88)
print(f"  검증 결과: {len(checks)-fails}/{len(checks)} PASS"
      + ("  ✓ 전 항목 통과" if fails == 0 else f"  ✗ {fails}건 실패"))
print("=" * 88)
sys.exit(1 if fails else 0)
