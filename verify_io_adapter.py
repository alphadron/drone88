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
   [P] 중복 KML   : Point+LineString 중복 KML(DJI Pilot 2 실물 내보내기 형태)
                     → 웨이포인트 중복 집계 방지 (실제 파사드 KML 테스트로 발견)
   [B] 촬영 경계   : Google Earth Pro 다각형 KML(사면 서쪽 절반만 포함)
                     → generate --boundary → 다각형 밖 웨이포인트 제외 확인
   [S] 경계→사면 자동구성 : 실측 topview 다각형 + 경사비만으로(3D 모델 없이)
                     기준선(PCA)·높이 자동 역산 → generate --slope-ratio 연동,
                     기준면이 다각형 전체를 덮는지(절반 누락 회귀 방지)·
                     flip_side 180° 반전 확인
 판정 항목 25건. 산출물: verify_io/ 폴더
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
from io_flightpath import load_flightpath
from io_model import build_slope_from_boundary

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

# ==================== [P] DJI 스타일 KML(Point+LineString 중복) → 파싱 ==========
print("\n[P] Point+LineString 중복 KML(DJI Pilot 2 실물 내보내기 형태) → 파싱 중복 방지")
# 실제 DJI 내보내기는 Waypoint(Placemark/Point) n개와 이를 잇는
# Wayline(Placemark/LineString, 동일 좌표 중복)을 같은 문서에 함께 담는다.
# 태그 구분 없이 모든 <coordinates>를 합치면 웨이포인트 수가 두 배로 잡힌다.
dup_pts = [(128.0, 37.0, 1.0), (128.0001, 37.0001, 2.0), (128.0002, 37.0002, 3.0)]
placemarks = "".join(
    f'<Placemark><Point><coordinates>{lo},{la},{al}</coordinates></Point></Placemark>'
    for lo, la, al in dup_pts)
line_coords = " ".join(f"{lo},{la},{al}" for lo, la, al in dup_pts)
dup_kml = os.path.join(OUT, "dup_point_line.kml")
os.makedirs(OUT, exist_ok=True)
with open(dup_kml, "w", encoding="utf-8") as f:
    f.write(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
        f'<Folder>{placemarks}</Folder>'
        f'<Placemark><LineString><coordinates>{line_coords}</coordinates></LineString></Placemark>'
        '</Document></kml>')
ip = load_flightpath(dup_kml, GeoAnchor(lat=37.0001, lon=128.0001, h=0.0), alt_mode="relative")
chk("P1 Point 우선·LineString 중복 배제", len(ip.enu) == len(dup_pts),
    f"파싱 {len(ip.enu)}개 (기대 {len(dup_pts)}개, Point+LineString 합산 시 {2*len(dup_pts)}개로 오염)")

# ==================== [B] 촬영 경계(Google Earth Pro 다각형) 제한 ==============
print("\n[B] 촬영 경계 다각형(Google Earth Pro 스타일 KML) → generate 범위 제한")
# 사면 서쪽 절반만 덮는 경계 다각형(투영좌표 X0 기준 서쪽) → lon/lat로 변환해 KML 저장
corners_xy = [(X0 - 70, Y0 - 70), (X0, Y0 - 70), (X0, Y0 + 70), (X0 - 70, Y0 + 70)]
corners_ll = [tr.transform(x, y) for x, y in corners_xy]
bnd_kml = os.path.join(OUT, "boundary.kml")
bk = simplekml.Kml()
bk.newpolygon(outerboundaryis=[(lo, la, 0.0) for lo, la in corners_ll])
bk.save(bnd_kml)

rc = run_cli("generate", obj, "--epsg", str(EPSG), "--gsd", "0.5", "--slope-min", "10",
             "--boundary", bnd_kml, "--out", os.path.join(OUT, "b"))
sb = load_summary(os.path.join(OUT, "b")) if rc == 0 else {}
chk("B1 경계 제한 파이프라인 완주", rc == 0, f"rc={rc}, WP {sb.get('n_waypoints')}")
chk("B2 경계 적용 후 웨이포인트 감소", rc == 0 and 0 < sb.get("n_waypoints", 0) < so["n_waypoints"],
    f"{sb.get('n_waypoints')}개 < 무제한 {so['n_waypoints']}개")
bcsv = os.path.join(OUT, "b", "mission_waypoints.csv")
e_vals = np.loadtxt(bcsv, delimiter=",", skiprows=1, usecols=3, encoding="utf-8-sig")
chk("B3 전 웨이포인트가 경계(서쪽 절반) 내부", rc == 0 and float(np.max(e_vals)) < 1.0,
    f"E 최대값 {float(np.max(e_vals)):.2f} m < 0(경계 동쪽 한계) 부근")

# ============ [S] 경계 다각형 기반 사면 자동 구성(기준선 연동) =================
print("\n[S] 경계 다각형(Google Earth Pro 실측 topview) → 사면 자동 구성·연동")
# S1: 함수 단위 검증 — 남북으로 긴 직사각형(bnd_kml과 동일 형상, 동서 폭
# 70m)을 로컬 ENU로 직접 만들어, 기준선(남북, PCA)·수평런(전체 폭 70m —
# 예전엔 중심선 기준 절반(35m)만 반영해 기준면이 실제 폭의 절반만 덮는
# 결함이 있었음, 실측 스크린샷으로 발견해 전체 폭 기준으로 수정)·역산
# 높이가 경계 폭에 정확히 연동되는지 확인
rect_en = np.array([[-35.0, -70.0], [35.0, -70.0], [35.0, 70.0], [-35.0, 70.0]])
mesh_s, info_s = build_slope_from_boundary(rect_en, 1.0, 0.3)
exp_alpha = np.degrees(np.arctan(1.0 / 0.3))
exp_run, exp_height = 70.0, 70.0 * (1.0 / 0.3)
chk("S1 기준선(PCA) 남북 자동검출", abs(abs(info_s["baseline_azimuth_deg"] % 180) - 0) < 1.0,
    f"기준선 방위 {info_s['baseline_azimuth_deg']:.1f}° (남북 0/180° 기대)")
chk("S2 경사각 1:0.3 → α=73.3°", abs(info_s["alpha_deg"] - exp_alpha) < 0.1,
    f"α={info_s['alpha_deg']:.2f}° (기대 {exp_alpha:.2f}°)")
chk("S3 경계 전체폭↔수평런 연동", abs(info_s["run_m"] - exp_run) < 0.1,
    f"run={info_s['run_m']:.2f} m (경계 전체폭 {exp_run} m와 일치)")
chk("S4 수평런↔높이 자동 역산", abs(info_s["height_m"] - exp_height) < 0.5,
    f"height={info_s['height_m']:.2f} m (기대 {exp_height:.2f} m)")

# S7: 메시 footprint가 다각형 전체를 덮는지(절반만 덮던 결함 회귀 방지) —
# 다각형 각 정점이 메시의 (E,N) 바운딩박스 안에 들어오는지 확인
mesh_bb_lo, mesh_bb_hi = mesh_s.vertices[:, :2].min(axis=0), mesh_s.vertices[:, :2].max(axis=0)
covers_all = bool(np.all(rect_en >= mesh_bb_lo - 1e-6) and np.all(rect_en <= mesh_bb_hi + 1e-6))
chk("S7 기준면이 다각형 전체를 커버(절반 누락 회귀 방지)", covers_all,
    f"다각형 범위 {rect_en.min(axis=0)}~{rect_en.max(axis=0)} ⊂ 메시 범위 "
    f"{mesh_bb_lo}~{mesh_bb_hi}")

# S8: flip_side로 사면 방향(aspect)이 정확히 180° 반전되는지
_, info_flip = build_slope_from_boundary(rect_en, 1.0, 0.3, flip_side=True)
aspect_diff = abs((info_s["aspect_deg"] - info_flip["aspect_deg"]) % 360 - 180)
chk("S8 flip_side로 사면방향 180° 반전", aspect_diff < 0.1,
    f"aspect {info_s['aspect_deg']:.1f}° -> flip {info_flip['aspect_deg']:.1f}°")

# S5~: CLI 통합 — input(3D모델) 없이 --boundary + --slope-ratio 만으로 generate
rc = run_cli("generate", "--slope-ratio", "1:0.3", "--boundary", bnd_kml,
             "--gsd", "0.5", "--slope-min", "60", "--out", os.path.join(OUT, "s"))
ssum = load_summary(os.path.join(OUT, "s")) if rc == 0 else {}
chk("S5 input 없이 경계+경사비만으로 생성", rc == 0, f"rc={rc}, WP {ssum.get('n_waypoints')}")
chk("S6 α=73.3°, φ=-16.7° 반영", rc == 0 and
    abs(ssum.get("surface", {}).get("slope_deg", 0) - exp_alpha) < 0.5 and
    max(abs(p - (exp_alpha - 90)) for p in ssum.get("pitch_range_deg", [999])) < 0.5,
    f"α={ssum.get('surface',{}).get('slope_deg')}°, φ 범위 {ssum.get('pitch_range_deg')}")

# ── 결과 ──
fails = sum(not ok for _, ok, _ in checks)
print("\n" + "=" * 88)
print(f"  검증 결과: {len(checks)-fails}/{len(checks)} PASS"
      + ("  ✓ 전 항목 통과" if fails == 0 else f"  ✗ {fails}건 실패"))
print("=" * 88)
sys.exit(1 if fails else 0)
