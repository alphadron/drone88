#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 camera.py — FacilityPath v1.0 / M2~M3 공용 모듈
 카메라 프리셋 · GSD↔접근거리 환산 · 면상 풋프린트/격자 간격 산출
--------------------------------------------------------------------------------
 (주)드론아이디 AI연구소 · 문서번호 DID-DP-2026-0829 부속
--------------------------------------------------------------------------------
 법선 정렬 촬영이므로 GSD 기준은 "고도"가 아닌 "면까지의 사거리 d"이다.
   d = GSD × f / p,  GSD = d × p / f
 촬영 자세는 가로(landscape) 기준: 장변(px_long)을 비행선 직교 방향(라인 간격)으로,
 단변(px_short)을 비행 진행 방향(트리거 간격)으로 배치한다. (orientation으로 전환 가능)
================================================================================
"""

from dataclasses import dataclass


@dataclass
class CameraSpec:
    name: str
    px_long: int            # 장변 픽셀
    px_short: int           # 단변 픽셀
    pixel_pitch_um: float   # 픽셀 피치 [µm]
    focal_mm: float         # 초점거리 [mm]
    min_interval_s: float   # 최소 촬영주기 [s]
    shutter_s: float        # 기준 셔터속도 [s]
    mb_per_photo: float     # 1매당 용량 [MB]

    # ── GSD ↔ 접근거리 ──
    def dist_for_gsd(self, gsd_m: float) -> float:
        return gsd_m * (self.focal_mm * 1e-3) / (self.pixel_pitch_um * 1e-6)

    def gsd_for_dist(self, dist_m: float) -> float:
        return dist_m * (self.pixel_pitch_um * 1e-6) / (self.focal_mm * 1e-3)

    # ── 면상 풋프린트 [m] ──
    def footprint(self, gsd_m: float, long_side_cross: bool = True) -> tuple:
        """(진행방향, 직교방향) 풋프린트. long_side_cross=True: 장변을 라인 직교로."""
        long_m, short_m = self.px_long * gsd_m, self.px_short * gsd_m
        return (short_m, long_m) if long_side_cross else (long_m, short_m)

    # ── 격자 간격 [m] ──
    def spacing(self, gsd_m: float, fwd_overlap: float, side_overlap: float,
                long_side_cross: bool = True) -> tuple:
        """(트리거 간격, 라인 간격)."""
        fp_along, fp_cross = self.footprint(gsd_m, long_side_cross)
        return fp_along * (1 - fwd_overlap), fp_cross * (1 - side_overlap)


# ── 프리셋 (JSON 외부화 전 단계의 내장 기본값) ─────────────────────────────────
PRESETS = {
    "P1_50mm": CameraSpec("DJI Zenmuse P1 (50mm)", 8192, 5460, 4.4, 50.0,
                          0.7, 1 / 1000, 30.0),
    "P1_35mm": CameraSpec("DJI Zenmuse P1 (35mm)", 8192, 5460, 4.4, 35.0,
                          0.7, 1 / 1000, 30.0),
    "P1_24mm": CameraSpec("DJI Zenmuse P1 (24mm)", 8192, 5460, 4.4, 24.0,
                          0.7, 1 / 1000, 30.0),
}


def get_camera(key: str = "P1_50mm") -> CameraSpec:
    if key not in PRESETS:
        raise KeyError(f"미등록 카메라 프리셋: {key} (등록: {list(PRESETS)})")
    return PRESETS[key]


if __name__ == "__main__":
    cam = get_camera()
    for gsd_cm in (0.2, 0.5):
        g = gsd_cm / 100
        d = cam.dist_for_gsd(g)
        trig, line = cam.spacing(g, 0.80, 0.70)
        print(f"GSD {gsd_cm} cm → d={d:.2f} m, 트리거 {trig:.2f} m, 라인 {line:.2f} m")
