#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 tilt_solver.py — FacilityPath v1.0 / M2 코어 모듈
 경사면 법선 정렬 짐벌 피치(φ)·헤딩(ψ) 및 촬영 웨이포인트 산출
--------------------------------------------------------------------------------
 (주)드론아이디 AI연구소 · 문서번호 DID-DP-2026-0829 부속 (M2)
--------------------------------------------------------------------------------
 핵심 공식 (개발계획서 §4.1):
   짐벌 피치  φ = α − 90°   [deg]  (수평 0°, 연직하방 −90°)
     · 평지   α= 0° → φ=−90° (연직)
     · 사면   α=60° → φ=−30°
     · 수직벽 α=90° → φ=  0° (수평)
   헤딩      ψ = azimuth(−n_h)  — 카메라는 법선 반대 방향(면 안쪽)을 본다
             = (aspect + 180°) mod 360
   웨이포인트 위치 = 면상 격자점 + n × d  (d: 접근거리)
 평활화:
   법선 이동평균(윈도 w) → 피치 변화율 제한(max_pitch_rate °/점)
================================================================================
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree


# ──────────────────────────────────────────────────────────────────────────────
# 결과 컨테이너
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Waypoint:
    position: np.ndarray        # 드론 위치 (면상점 + n·d) [m, ENU]
    surface_point: np.ndarray   # 대응 면상점 [m]
    normal: np.ndarray          # 국부 법선 (상향, 평활화 후)
    gimbal_pitch_deg: float     # φ [deg]
    heading_deg: float          # ψ [deg, 0~360]
    slope_deg: float            # 국부 경사각 α [deg]


@dataclass
class TiltSolution:
    waypoints: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)   # pitch min/max/mean 등


# ──────────────────────────────────────────────────────────────────────────────
# 솔버
# ──────────────────────────────────────────────────────────────────────────────
class TiltSolver:
    """
    면상 샘플점 시퀀스에 대해 국부 법선 → 짐벌 피치·헤딩 → 웨이포인트를 산출.

    Parameters
    ----------
    approach_dist_m : 접근거리 d (면 법선 방향 이격) [m]
    fit_radius_m    : 국부 법선 추정용 이웃 반경 [m] (평면 SVD 피팅)
    smooth_window   : 법선 이동평균 윈도 (홀수 권장, 1=비활성)
    max_pitch_rate  : 인접 웨이포인트 간 피치 변화 상한 [deg] (0=비활성)
    gimbal_range    : 기체 짐벌 가동범위 (min, max) [deg] — 위반 시 경고 수집
    """

    def __init__(self, approach_dist_m: float = 15.0, fit_radius_m: float = 3.0,
                 smooth_window: int = 3, max_pitch_rate: float = 10.0,
                 gimbal_range: tuple = (-120.0, 30.0)):
        self.d = approach_dist_m
        self.r = fit_radius_m
        self.w = max(1, int(smooth_window))
        self.rate = max_pitch_rate
        self.gmin, self.gmax = gimbal_range
        self.warnings: list = []

    # ── ① 국부 법선 추정 ──
    def local_normals(self, sample_pts: np.ndarray, mesh_vertices: np.ndarray,
                      fallback_normal: np.ndarray) -> np.ndarray:
        """각 샘플점 주변 정점(반경 r)에 SVD 평면 피팅 → 상향 법선."""
        tree = cKDTree(mesh_vertices)
        normals = np.zeros_like(sample_pts)
        for i, p in enumerate(sample_pts):
            idx = tree.query_ball_point(p, self.r)
            if len(idx) < 3:                       # 이웃 부족 → 면 평균 법선 대체
                normals[i] = fallback_normal
                continue
            q = mesh_vertices[idx] - mesh_vertices[idx].mean(axis=0)
            # SVD 최소 특이벡터 = 평면 법선
            n = np.linalg.svd(q, full_matrices=False)[2][-1]
            if n[2] < 0:                           # 상향 정규화
                n = -n
            if n[2] == 0 and np.dot(n, fallback_normal) < 0:  # 수직벽 방향 일관화
                n = -n
            normals[i] = n / np.linalg.norm(n)
        return normals

    # ── ② 평활화 ──
    def smooth_normals(self, normals: np.ndarray) -> np.ndarray:
        if self.w <= 1 or len(normals) < 3:
            return normals
        k = self.w
        pad = k // 2
        ext = np.vstack([normals[:1].repeat(pad, 0), normals,
                         normals[-1:].repeat(pad, 0)])
        kern = np.ones(k) / k
        sm = np.column_stack([np.convolve(ext[:, c], kern, mode="valid")
                              for c in range(3)])
        sm /= np.linalg.norm(sm, axis=1, keepdims=True)
        return sm

    # ── ③ 법선 → 피치·헤딩 (핵심 공식) ──
    @staticmethod
    def pitch_heading_from_normal(n: np.ndarray) -> tuple:
        """
        상향 법선 n → (φ, ψ, α).
        φ = α − 90°,  ψ = azimuth(−n_h) = (aspect + 180°) mod 360.
        평지(수평성분≈0)는 ψ=NaN (임무 설계 단계에서 진행방향으로 대체).
        """
        nz = float(np.clip(n[2], -1.0, 1.0))
        alpha = np.degrees(np.arccos(nz))          # 경사각
        phi = alpha - 90.0                         # ★ 짐벌 피치
        if np.hypot(n[0], n[1]) < 1e-9:
            psi = float("nan")
        else:
            psi = float(np.degrees(np.arctan2(-n[0], -n[1])) % 360.0)
        return phi, psi, alpha

    # ── ④ 피치 변화율 제한 ──
    def limit_pitch_rate(self, pitch: np.ndarray) -> np.ndarray:
        if self.rate <= 0 or len(pitch) < 2:
            return pitch
        out = pitch.copy()
        for i in range(1, len(out)):
            dlt = np.clip(out[i] - out[i - 1], -self.rate, self.rate)
            out[i] = out[i - 1] + dlt
        return out

    # ── 통합 실행 ──
    def solve(self, sample_pts: np.ndarray, mesh_vertices: np.ndarray,
              fallback_normal: np.ndarray) -> TiltSolution:
        self.warnings.clear()
        n_raw = self.local_normals(sample_pts, mesh_vertices, fallback_normal)
        n_sm = self.smooth_normals(n_raw)

        phis, psis, alphas = [], [], []
        for n in n_sm:
            phi, psi, alpha = self.pitch_heading_from_normal(n)
            phis.append(phi); psis.append(psi); alphas.append(alpha)
        phis = self.limit_pitch_rate(np.array(phis))

        sol = TiltSolution()
        for p, n, phi, psi, alpha in zip(sample_pts, n_sm, phis, psis, alphas):
            if not (self.gmin <= phi <= self.gmax):
                self.warnings.append(
                    f"[짐벌 한계] φ={phi:.1f}° at {np.round(p,1)} "
                    f"(허용 {self.gmin}~{self.gmax}°)")
            sol.waypoints.append(Waypoint(
                position=p + n * self.d, surface_point=p, normal=n,
                gimbal_pitch_deg=float(phi), heading_deg=float(psi),
                slope_deg=float(alpha)))
        sol.stats = dict(
            n=len(sol.waypoints),
            pitch_min=float(phis.min()), pitch_max=float(phis.max()),
            pitch_mean=float(phis.mean()),
            slope_mean=float(np.mean(alphas)),
            n_warnings=len(self.warnings))
        return sol
