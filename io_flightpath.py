#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 io_flightpath.py — FacilityPath v1.1 / [1'단계] 기존 비행경로 입력
 UgCS·DJI Pilot 2 등에서 만든 경로(KML / WPML .kmz / CSV) → 로컬 ENU 웨이포인트
--------------------------------------------------------------------------------
 (주)드론아이디 AI연구소 · DID-DP-2026-0829 부속 (2026-09-05)
--------------------------------------------------------------------------------
 지원 형식:
   .kml  : LineString coordinates 또는 Placemark/Point (UgCS·Google Earth 내보내기)
   .kmz  : DJI WPML (wpmz/waylines.wpml — Point + executeHeight)
   .csv  : 열 lon,lat,alt  또는 E,N,U (헤더로 자동 판별)
 고도 규약(alt_mode):
   "absolute" : 타원체고/해발 → anchor.h 를 빼서 ENU z
   "relative" : 이륙점 상대고도 → takeoff_z + alt 를 ENU z 로 사용
================================================================================
"""

import csv
import os
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from kml_writer import GeoAnchor, EnuConverter, KML_NS, WPML_NS

NS = {"k": KML_NS, "w": WPML_NS}


@dataclass
class InputPath:
    enu: np.ndarray            # (N,3) 로컬 ENU 웨이포인트 (원 순서 유지)
    source: str
    fmt: str                   # kml | wpml | csv
    note: str = ""


def _parse_kml_coords(root) -> np.ndarray:
    """Placemark/Point 좌표를 우선 사용하고, Point가 하나도 없을 때만
    Placemark/LineString 좌표로 대체한다. DJI Pilot 2류 KML은 개별
    Waypoint(Point)와 이를 잇는 Wayline(LineString)을 같은 문서에 함께
    내보내는데, 태그 구분 없이 모든 coordinates를 합치면 동일 경로가
    두 배로 중복 집계된다."""
    def _coords_text(placemark):
        for c in placemark.iter():
            if c.tag.endswith("coordinates") and c.text:
                return c.text
        return None

    def _tokens(text):
        out = []
        for tok in text.split():
            v = [float(x) for x in tok.split(",")]
            out.append(v + [0.0] * (3 - len(v)))
        return out

    point_pts, line_pts = [], []
    for pm in root.iter():
        if not pm.tag.endswith("Placemark"):
            continue
        text = _coords_text(pm)
        if not text:
            continue
        kind_tags = {c.tag.rsplit("}", 1)[-1] for c in pm.iter()}
        (point_pts if "Point" in kind_tags else line_pts).extend(_tokens(text))

    pts = point_pts or line_pts
    if not pts:
        raise ValueError("KML에 coordinates 요소가 없습니다")
    return np.array(pts)


def _read_kml(path: str) -> np.ndarray:
    return _parse_kml_coords(ET.parse(path).getroot())


def _read_wpml(path: str) -> np.ndarray:
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith("waylines.wpml")]
        if not names:
            raise ValueError("KMZ 안에 waylines.wpml이 없습니다")
        root = ET.fromstring(z.read(names[0]))
    pts = []
    for pm in root.iter():
        if not pm.tag.endswith("Placemark"):
            continue
        c = pm.find(".//k:coordinates", NS)
        h = pm.find(".//w:executeHeight", NS)
        if c is None:
            continue
        lon, lat = [float(x) for x in c.text.strip().split(",")[:2]]
        pts.append([lon, lat, float(h.text) if h is not None else 0.0])
    return np.array(pts)


def _read_csv(path: str):
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    hdr = [h.strip().lower() for h in rows[0]]
    data = np.array([[float(x) for x in r[:3]] for r in rows[1:] if r and r[0]])
    enu_names = ("e", "x", "east", "n", "y", "north", "u", "z", "up")
    if any(k in hdr[0] for k in ("lon", "lng", "경도")):
        is_geo = True
    elif hdr[0] in enu_names:
        is_geo = False
    else:                                        # 헤더 힌트 없음 → 값 범위로 판별
        is_geo = abs(data[:, 0]).max() <= 180 and abs(data[:, 1]).max() <= 90
    return data, is_geo


def load_flightpath(path: str, anchor: GeoAnchor, alt_mode: str = "relative",
                    takeoff_z: float = 0.0) -> InputPath:
    """기존 경로 파일 → 로컬 ENU 웨이포인트."""
    ext = os.path.splitext(path)[1].lower()
    conv = EnuConverter(anchor)

    if ext == ".kml":
        llh, fmt = _read_kml(path), "kml"
    elif ext == ".kmz":
        llh, fmt = _read_wpml(path), "wpml"
        alt_mode = "relative"                      # WPML executeHeight는 이륙점 상대
    elif ext == ".csv":
        data, is_geo = _read_csv(path)
        if not is_geo:                             # 이미 ENU
            return InputPath(data, path, "csv", "ENU 직접 입력")
        llh, fmt = data, "csv"
    else:
        raise ValueError(f"지원하지 않는 경로 형식: {ext}")

    if alt_mode == "relative":
        # 평면 변환만 하고 z는 상대고도 그대로(이륙점 기준)
        enu = conv.from_wgs84(np.column_stack([llh[:, 0], llh[:, 1],
                                               np.full(len(llh), anchor.h)]))
        enu[:, 2] = takeoff_z + llh[:, 2]
    else:
        enu = conv.from_wgs84(llh)
    return InputPath(enu, path, fmt, f"alt_mode={alt_mode}, n={len(enu)}")
