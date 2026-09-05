#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 verify_kml_writer.py — FacilityPath M4 출력 계층 자동 검증
--------------------------------------------------------------------------------
 (주)드론아이디 AI연구소 · 문서번호 DID-DP-2026-0829 부속 (M4)
--------------------------------------------------------------------------------
 파이프라인: M3 SlopePlanner 재실행(2소티 강제) → 실제 현장 기준점으로 지오레퍼런싱
             → KML + 소티별 WPML(.kmz) 출력 → 구조·수치 자동 검증
 기준점: 절토사면 현장 (36°13′27.73″N, 127°16′35.62″E, 타원체고 121 m 가정)
 검증 항목:
   [K1] ENU↔WGS84 왕복 평면오차 ≤ 1 mm (AEQD 지역투영 무결성)
   [K2] KML 파싱 유효 + 좌표 lon,lat 순서·현장 좌표 근방(±0.02°) 확인
   [K3] KML 고도 = 기준점 타원체고 + ENU z (absolute 모드) 일치 ≤ 1 cm
   [K4] KMZ 구조: wpmz/template.kml + wpmz/waylines.wpml 존재, XML 유효
   [K5] WPML 인덱스 연속성(0..N−1), 소티별 WP 수 합계 = 전체
   [K6] WPML 짐벌 피치 = α−90° (±0.1°), 헤딩 ∈ [−180,180] & 기대값 일치
   [K7] executeHeight = ENU z − 이륙점 z 일치 ≤ 1 cm
 산출물: slope_mission_review.kml, slope_mission_sortie1.kmz, _sortie2.kmz
================================================================================
"""

import sys
import zipfile
import xml.etree.ElementTree as ET

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np

from camera import get_camera
from surface_analyzer import SurfaceAnalyzer, make_synthetic_slope
from planner_base import PlanConfig
from planner_slope import SlopePlanner
from kml_writer import GeoAnchor, EnuConverter, export_mission, WPML_NS, KML_NS

# 현장 기준점 (UgCS 캡처 좌표)
ANCHOR = GeoAnchor(lat=36.0 + 13/60 + 27.73/3600,
                   lon=127.0 + 16/60 + 35.62/3600, h=121.0)
ALPHA, ASPECT = 60.0, 180.0
TAKEOFF_Z = 0.0
NSK = {"k": KML_NS, "w": WPML_NS}


def build():
    mesh = make_synthetic_slope(ALPHA, ASPECT, width_m=120, length_m=40, grid=40)
    surf = SurfaceAnalyzer(slope_min_deg=10, min_area_m2=10).analyze(mesh).surfaces[0]
    cfg = PlanConfig(gsd_m=0.005, speed_ms=2.5, safe_endurance_min=3.0,  # 2소티 강제
                     max_pitch_rate=0.0)
    res = SlopePlanner(get_camera("P1_50mm"), cfg).plan(surf, mesh.vertices)
    return res


def main():
    print("=" * 88)
    print("  FacilityPath M4 검증 — kml_writer (Google Earth KML + DJI Pilot 2 WPML)")
    print(f"  기준점: {ANCHOR.lat:.7f}N, {ANCHOR.lon:.7f}E, h={ANCHOR.h} m / "
          f"사면 α={ALPHA:.0f}° → φ={ALPHA-90:.0f}°")
    print("=" * 88)

    res = build()
    files, rt_err = export_mission(res.waypoints, res.sortie_index, ANCHOR,
                                   out_prefix="slope_mission",
                                   speed_ms=2.5, takeoff_z_m=TAKEOFF_Z)
    print(f"\n  생성 파일: {', '.join(files)}")
    print(f"  소티 구성: {res.n_sorties}개 / 웨이포인트 {len(res.waypoints)}개")

    checks = []
    enu = res.positions()

    # ── [K1] 왕복오차 ──
    checks.append(("K1 ENU↔WGS84 왕복오차", rt_err <= 1e-3,
                   f"{rt_err*1000:.4f} mm ≤ 1 mm"))

    # ── [K2][K3] KML 검사 ──
    tree = ET.parse("slope_mission_review.kml")
    coords_el = tree.findall(".//k:LineString/k:coordinates", NSK)
    pts = []
    for el in coords_el:
        for tok in el.text.split():
            pts.append([float(x) for x in tok.split(",")])
    pts = np.array(pts)
    lon_ok = np.all(np.abs(pts[:, 0] - ANCHOR.lon) < 0.02)
    lat_ok = np.all(np.abs(pts[:, 1] - ANCHOR.lat) < 0.02)
    checks.append(("K2 KML 좌표 순서·현장 근방", bool(lon_ok and lat_ok),
                   f"lon,lat 순서 ✓, Δlon<{np.abs(pts[:,0]-ANCHOR.lon).max():.4f}°"))
    alt_err = np.abs(np.sort(pts[:, 2]) - np.sort(ANCHOR.h + enu[:, 2])).max()
    checks.append(("K3 KML absolute 고도", alt_err <= 0.01,
                   f"기준점 h+z 대비 최대오차 {alt_err*100:.2f} cm"))

    # ── [K4]~[K7] WPML 검사 ──
    total_wp, all_pitch, all_head, all_h = 0, [], [], []
    struct_ok, idx_ok = True, True
    for s in range(res.n_sorties):
        kmz = f"slope_mission_sortie{s+1}.kmz"
        with zipfile.ZipFile(kmz) as z:
            names = set(z.namelist())
            struct_ok &= {"wpmz/template.kml", "wpmz/waylines.wpml"} <= names
            root = ET.fromstring(z.read("wpmz/waylines.wpml"))
        idxs = [int(e.text) for e in root.findall(".//w:index", NSK)]
        idx_ok &= idxs == list(range(len(idxs)))
        total_wp += len(idxs)
        all_pitch += [float(e.text) for e in
                      root.findall(".//w:gimbalPitchRotateAngle", NSK)]
        all_head += [float(e.text) for e in
                     root.findall(".//w:waypointHeadingAngle", NSK)]
        all_h += [float(e.text) for e in root.findall(".//w:executeHeight", NSK)]
    checks.append(("K4 KMZ 구조·XML 유효", struct_ok,
                   "wpmz/template.kml + waylines.wpml ✓"))
    checks.append(("K5 인덱스 연속·WP 총수", idx_ok and total_wp == len(res.waypoints),
                   f"소티별 0..N−1 ✓, 합계 {total_wp} = {len(res.waypoints)}"))

    all_pitch, all_head = np.array(all_pitch), np.array(all_head)
    pitch_err = np.abs(all_pitch - (ALPHA - 90.0)).max()
    head_exp = ((ASPECT + 180.0) + 180.0) % 360.0 - 180.0    # 남향→기대 0°
    head_ok = (np.all((all_head >= -180) & (all_head <= 180))
               and np.abs(all_head - head_exp).max() <= 0.5)
    checks.append(("K6 WPML 짐벌·헤딩", pitch_err <= 0.1 and head_ok,
                   f"φ 오차 {pitch_err:.3f}° / ψ={all_head.mean():.1f}° "
                   f"(기대 {head_exp:.0f}°), 범위 ±180 ✓"))

    h_err = np.abs(np.sort(np.array(all_h)) -
                   np.sort(enu[:, 2] - TAKEOFF_Z)).max()
    checks.append(("K7 executeHeight 상대고도", h_err <= 0.01,
                   f"ENU z−이륙점 대비 최대오차 {h_err*100:.2f} cm"))

    # ── 판정 ──
    print("\n  ── 검증 판정 ──")
    fails = 0
    for name, ok, detail in checks:
        fails += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<26s} {detail}")

    print("\n" + "=" * 88)
    print(f"  검증 결과: {len(checks)-fails}/{len(checks)} PASS"
          + ("  ✓ 전 항목 통과" if fails == 0 else f"  ✗ {fails}건 실패"))
    print("  ※ Pilot 2 실기 임포트 및 DJI 열거값(기체 60/페이로드 50) 대조는")
    print("     현장 PC에서 수행 — 상이 시 kml_writer.py 상단 상수만 교체 (§8 리스크)")
    print("=" * 88)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
