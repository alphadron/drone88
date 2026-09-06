#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 planner_slope.py — FacilityPath v1.0 / M3 전략 계층
 급경사지(절토사면) 등고선형 서펜타인 경로 생성 전략 (개발계획서 §4.3)
--------------------------------------------------------------------------------
 (주)드론아이디 AI연구소 · 문서번호 DID-DP-2026-0829 부속 (M3)
--------------------------------------------------------------------------------
 전략:
   · 촬영 라인 = 등고선 방향(u, 수평) 직선 — 고도 일정 유지로 조종·감시 용이
   · 라인 적층 = 경사 방향(v)으로 라인 간격(횡중복 반영)만큼 오프셋
   · 웨이포인트 = 라인 위 트리거 간격(종중복 반영) 등간격 배치
   · 헤딩 = 면을 정면으로 바라보는 방향 고정(크랩 비행) — TiltSolver가 산출
   · 라인 양끝은 margin_frac 만큼 연장하여 가장자리 커버리지 확보
 다단 사면(소단 분리)은 SurfaceAnalyzer가 면을 분리 추출하므로
 면별로 plan()을 반복 호출하면 된다 — 본 전략은 단일 면을 담당한다.
================================================================================
"""

import numpy as np

from planner_base import PathPlanner
from surface_analyzer import SurfaceInfo


class SlopePlanner(PathPlanner):
    """급경사지: 등고선형(수평 라인) 서펜타인."""

    def design_lines(self, surf: SurfaceInfo, trigger_m: float,
                     line_m: float) -> list:
        u, v = self.surface_axes(surf)
        m = self.cfg.margin_frac

        # 커버 범위 (면 중심 기준 ±절반 + 여유)
        half_u = surf.extent_u_m * (0.5 + m)
        half_v = surf.extent_v_m * (0.5 + m)

        # 경사(v) 방향 라인 위치: 간격 ≤ line_m 이 되도록 균등 분할
        n_lines = max(2, int(np.ceil(2 * half_v / line_m)) + 1)
        v_offsets = np.linspace(-half_v, half_v, n_lines)

        # 등고선(u) 방향 웨이포인트: 간격 ≤ trigger_m 균등 분할
        n_wp = max(2, int(np.ceil(2 * half_u / trigger_m)) + 1)
        u_offsets = np.linspace(-half_u, half_u, n_wp)

        # 하단 라인부터 상승 적층 (v_offsets 오름차순 = 아래→위)
        lines = []
        for vo in v_offsets:
            pts = surf.centroid + np.outer(u_offsets, u) + vo * v
            lines.append(pts)
        return lines
