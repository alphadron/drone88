#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 kml_writer.py — FacilityPath v1.0 / M4 출력 계층
 웨이포인트 → ① Google Earth 검토용 KML  ② DJI Pilot 2용 WPML(.kmz)
--------------------------------------------------------------------------------
 (주)드론아이디 AI연구소 · 문서번호 DID-DP-2026-0829 부속 (M4)
--------------------------------------------------------------------------------
 좌표 변환:
   로컬 ENU(m) → WGS84 경위도.  기준점(anchor lat/lon/h)에 지역 방위등거리
   투영(AEQD, WGS84 타원체)을 세워 pyproj로 정변환한다. 수백 m 규모 현장에서
   근사오차는 mm 미만이며, 역변환 왕복검사(verify)로 매 실행 시 확인한다.
 고도 규약:
   ENU z(기준점 상대높이) → WPML executeHeight (relativeToStartPoint, 이륙점
   기준 상대고도). 이륙점의 ENU z를 takeoff_z_m 로 지정한다(기본 0 = 기준점).
   KML은 absolute 고도(타원체고 anchor_h + z)로 기록해 Earth 지형과 비교한다.
 DJI WPML:
   KMZ 구조 = wpmz/template.kml + wpmz/waylines.wpml (네임스페이스 1.0.2).
   기체/페이로드 열거값 기본: M300 RTK=60, P1=50 — 개발계획서 §8 리스크 대응
   대로 Pilot 2 내보내기 샘플과 대조 검증할 것(상이 시 상수만 교체).
================================================================================
"""

import zipfile
from dataclasses import dataclass
from xml.sax.saxutils import escape

import numpy as np
import simplekml
from pyproj import CRS, Transformer

WPML_NS = "http://www.dji.com/wpmz/1.0.2"
KML_NS = "http://www.opengis.net/kml/2.2"

# DJI 열거값 기본치 (Pilot 2 샘플 대조 검증 대상 — §8 리스크)
DRONE_ENUM_M300 = 60
PAYLOAD_ENUM_P1 = 50

SORTIE_COLORS = ["ff85720b", "ffc27ba0", "ff2a8cc9", "ff2aa198",
                 "ff6c71c4", "ff268bd2"]          # KML aabbggrr


# ──────────────────────────────────────────────────────────────────────────────
# 좌표 변환기
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class GeoAnchor:
    lat: float          # 기준점 위도 [deg, WGS84]
    lon: float          # 기준점 경도 [deg]
    h: float = 0.0      # 기준점 타원체고 [m]


class EnuConverter:
    """로컬 ENU(m) ↔ WGS84. AEQD 지역투영 기반, 왕복검사 내장."""

    def __init__(self, anchor: GeoAnchor):
        self.anchor = anchor
        aeqd = CRS.from_proj4(
            f"+proj=aeqd +lat_0={anchor.lat} +lon_0={anchor.lon} "
            f"+ellps=WGS84 +units=m +no_defs")
        self.fwd = Transformer.from_crs(aeqd, CRS.from_epsg(4326), always_xy=True)
        self.inv = Transformer.from_crs(CRS.from_epsg(4326), aeqd, always_xy=True)

    def to_wgs84(self, enu: np.ndarray) -> np.ndarray:
        """(N,3) ENU → (N,3) [lon, lat, ellipsoidal h]."""
        lon, lat = self.fwd.transform(enu[:, 0], enu[:, 1])
        h = self.anchor.h + enu[:, 2]
        return np.column_stack([lon, lat, h])

    def from_wgs84(self, llh: np.ndarray) -> np.ndarray:
        """(N,3) [lon, lat, ellipsoidal h] → (N,3) ENU. (기존 경로 입력용 역변환)"""
        x, y = self.inv.transform(llh[:, 0], llh[:, 1])
        return np.column_stack([x, y, llh[:, 2] - self.anchor.h])

    def roundtrip_error_m(self, enu: np.ndarray) -> float:
        """정→역 왕복 최대 평면오차 [m] — 매 출력 시 무결성 검사용."""
        w = self.to_wgs84(enu)
        x, y = self.inv.transform(w[:, 0], w[:, 1])
        return float(np.max(np.hypot(x - enu[:, 0], y - enu[:, 1])))


# ──────────────────────────────────────────────────────────────────────────────
# ① Google Earth 검토용 KML
# ──────────────────────────────────────────────────────────────────────────────
def write_review_kml(waypoints, sortie_index, conv: EnuConverter, path: str,
                     name: str = "FacilityPath Mission", wp_label_step: int = 5):
    """
    소티별 경로 라인 + 웨이포인트 포인트(짐벌·헤딩 ExtendedData) KML.
    고도 absolute(타원체고) — Earth 지형 대비 이격 확인용.
    """
    enu = np.array([w.position for w in waypoints])
    llh = conv.to_wgs84(enu)
    kml = simplekml.Kml(name=name)

    for s in np.unique(sortie_index):
        sel = np.where(sortie_index == s)[0]
        fol = kml.newfolder(name=f"Sortie {s + 1}")
        ls = fol.newlinestring(name=f"Path S{s + 1}",
                               coords=[tuple(llh[i]) for i in sel])
        ls.altitudemode = simplekml.AltitudeMode.absolute
        ls.style.linestyle.color = SORTIE_COLORS[int(s) % len(SORTIE_COLORS)]
        ls.style.linestyle.width = 3

        for k, i in enumerate(sel):
            w = waypoints[i]
            if k % wp_label_step and k != len(sel) - 1:   # 라벨 과밀 방지
                continue
            p = fol.newpoint(name=f"WP{i}", coords=[tuple(llh[i])])
            p.altitudemode = simplekml.AltitudeMode.absolute
            p.style.iconstyle.scale = 0.5
            p.style.iconstyle.icon.href = (
                "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png")
            ed = p.extendeddata
            ed.newdata("gimbal_pitch_deg", f"{w.gimbal_pitch_deg:.2f}")
            ed.newdata("heading_deg", f"{w.heading_deg:.2f}")
            ed.newdata("slope_deg", f"{w.slope_deg:.2f}")
            ed.newdata("rel_height_m", f"{w.position[2]:.2f}")
    kml.save(path)
    return llh


# ──────────────────────────────────────────────────────────────────────────────
# ② DJI Pilot 2용 WPML (.kmz)
# ──────────────────────────────────────────────────────────────────────────────
def _heading_pm180(h_deg: float, travel_dir=None) -> float:
    """0~360 → −180~180. NaN(평지 등)은 진행방향 방위각으로 대체."""
    if np.isnan(h_deg):
        if travel_dir is None:
            return 0.0
        h_deg = float(np.degrees(np.arctan2(travel_dir[0], travel_dir[1])) % 360)
    return ((h_deg + 180.0) % 360.0) - 180.0


def _wpml_placemark(idx, lon, lat, exec_h, speed, heading, pitch) -> str:
    return f"""    <Placemark>
      <Point><coordinates>{lon:.9f},{lat:.9f}</coordinates></Point>
      <wpml:index>{idx}</wpml:index>
      <wpml:executeHeight>{exec_h:.2f}</wpml:executeHeight>
      <wpml:waypointSpeed>{speed:.1f}</wpml:waypointSpeed>
      <wpml:waypointHeadingParam>
        <wpml:waypointHeadingMode>smoothTransition</wpml:waypointHeadingMode>
        <wpml:waypointHeadingAngle>{heading:.1f}</wpml:waypointHeadingAngle>
        <wpml:waypointPoiPoint>0.000000,0.000000,0.000000</wpml:waypointPoiPoint>
        <wpml:waypointHeadingAngleEnable>1</wpml:waypointHeadingAngleEnable>
        <wpml:waypointHeadingPathMode>followBadArc</wpml:waypointHeadingPathMode>
      </wpml:waypointHeadingParam>
      <wpml:waypointTurnParam>
        <wpml:waypointTurnMode>toPointAndStopWithDiscontinuityCurvature</wpml:waypointTurnMode>
        <wpml:waypointTurnDampingDist>0</wpml:waypointTurnDampingDist>
      </wpml:waypointTurnParam>
      <wpml:useStraightLine>1</wpml:useStraightLine>
      <wpml:actionGroup>
        <wpml:actionGroupId>{idx}</wpml:actionGroupId>
        <wpml:actionGroupStartIndex>{idx}</wpml:actionGroupStartIndex>
        <wpml:actionGroupEndIndex>{idx}</wpml:actionGroupEndIndex>
        <wpml:actionGroupMode>sequence</wpml:actionGroupMode>
        <wpml:actionTrigger><wpml:actionTriggerType>reachPoint</wpml:actionTriggerType></wpml:actionTrigger>
        <wpml:action>
          <wpml:actionId>0</wpml:actionId>
          <wpml:actionActuatorFunc>gimbalRotate</wpml:actionActuatorFunc>
          <wpml:actionActuatorFuncParam>
            <wpml:gimbalRotateMode>absoluteAngle</wpml:gimbalRotateMode>
            <wpml:gimbalPitchRotateEnable>1</wpml:gimbalPitchRotateEnable>
            <wpml:gimbalPitchRotateAngle>{pitch:.1f}</wpml:gimbalPitchRotateAngle>
            <wpml:gimbalRollRotateEnable>0</wpml:gimbalRollRotateEnable>
            <wpml:gimbalRollRotateAngle>0</wpml:gimbalRollRotateAngle>
            <wpml:gimbalYawRotateEnable>0</wpml:gimbalYawRotateEnable>
            <wpml:gimbalYawRotateAngle>0</wpml:gimbalYawRotateAngle>
            <wpml:gimbalRotateTimeEnable>0</wpml:gimbalRotateTimeEnable>
            <wpml:gimbalRotateTime>0</wpml:gimbalRotateTime>
            <wpml:payloadPositionIndex>0</wpml:payloadPositionIndex>
          </wpml:actionActuatorFuncParam>
        </wpml:action>
        <wpml:action>
          <wpml:actionId>1</wpml:actionId>
          <wpml:actionActuatorFunc>takePhoto</wpml:actionActuatorFunc>
          <wpml:actionActuatorFuncParam>
            <wpml:payloadPositionIndex>0</wpml:payloadPositionIndex>
          </wpml:actionActuatorFuncParam>
        </wpml:action>
      </wpml:actionGroup>
    </Placemark>"""


def _mission_config(speed) -> str:
    return f"""  <wpml:missionConfig>
    <wpml:flyToWaylineMode>safely</wpml:flyToWaylineMode>
    <wpml:finishAction>goHome</wpml:finishAction>
    <wpml:exitOnRCLost>executeLostAction</wpml:exitOnRCLost>
    <wpml:executeRCLostAction>goBack</wpml:executeRCLostAction>
    <wpml:takeOffSecurityHeight>20</wpml:takeOffSecurityHeight>
    <wpml:globalTransitionalSpeed>{speed:.1f}</wpml:globalTransitionalSpeed>
    <wpml:droneInfo>
      <wpml:droneEnumValue>{DRONE_ENUM_M300}</wpml:droneEnumValue>
      <wpml:droneSubEnumValue>0</wpml:droneSubEnumValue>
    </wpml:droneInfo>
    <wpml:payloadInfo>
      <wpml:payloadEnumValue>{PAYLOAD_ENUM_P1}</wpml:payloadEnumValue>
      <wpml:payloadSubEnumValue>0</wpml:payloadSubEnumValue>
      <wpml:payloadPositionIndex>0</wpml:payloadPositionIndex>
    </wpml:payloadInfo>
  </wpml:missionConfig>"""


def write_wpml_kmz(waypoints, conv: EnuConverter, path: str,
                   speed_ms: float = 2.5, takeoff_z_m: float = 0.0,
                   mission_name: str = "FacilityPath"):
    """
    단일 소티 웨이포인트 목록 → Pilot 2 임포트용 .kmz
    (wpmz/template.kml + wpmz/waylines.wpml).
    executeHeight = ENU z − takeoff_z_m (relativeToStartPoint).
    """
    enu = np.array([w.position for w in waypoints])
    llh = conv.to_wgs84(enu)

    pms = []
    for i, w in enumerate(waypoints):
        trv = None
        if i + 1 < len(enu):
            trv = enu[i + 1] - enu[i]
        elif i > 0:
            trv = enu[i] - enu[i - 1]
        pms.append(_wpml_placemark(
            i, llh[i, 0], llh[i, 1], enu[i, 2] - takeoff_z_m, speed_ms,
            _heading_pm180(w.heading_deg, trv), w.gimbal_pitch_deg))

    waylines = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="{KML_NS}" xmlns:wpml="{WPML_NS}">
<Document>
{_mission_config(speed_ms)}
  <Folder>
    <wpml:templateId>0</wpml:templateId>
    <wpml:executeHeightMode>relativeToStartPoint</wpml:executeHeightMode>
    <wpml:waylineId>0</wpml:waylineId>
    <wpml:autoFlightSpeed>{speed_ms:.1f}</wpml:autoFlightSpeed>
{chr(10).join(pms)}
  </Folder>
</Document>
</kml>
"""
    template = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="{KML_NS}" xmlns:wpml="{WPML_NS}">
<Document>
  <wpml:author>DroneID FacilityPath</wpml:author>
  <wpml:createTime>0</wpml:createTime>
  <wpml:updateTime>0</wpml:updateTime>
{_mission_config(speed_ms)}
  <Folder>
    <wpml:templateType>waypoint</wpml:templateType>
    <wpml:templateId>0</wpml:templateId>
    <wpml:waylineCoordinateSysParam>
      <wpml:coordinateMode>WGS84</wpml:coordinateMode>
      <wpml:heightMode>relativeToStartPoint</wpml:heightMode>
    </wpml:waylineCoordinateSysParam>
    <wpml:autoFlightSpeed>{speed_ms:.1f}</wpml:autoFlightSpeed>
  </Folder>
</Document>
</kml>
"""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("wpmz/template.kml", template)
        z.writestr("wpmz/waylines.wpml", waylines)
    return llh


def export_mission(waypoints, sortie_index, anchor: GeoAnchor,
                   out_prefix: str = "mission", speed_ms: float = 2.5,
                   takeoff_z_m: float = 0.0):
    """
    통합 내보내기:
      {prefix}_review.kml            — 전 소티 Earth 검토용
      {prefix}_sortie{N}.kmz         — 소티별 Pilot 2 WPML
    반환: (생성 파일 목록, ENU 왕복오차[m])
    """
    conv = EnuConverter(anchor)
    enu = np.array([w.position for w in waypoints])
    rt_err = conv.roundtrip_error_m(enu)

    files = []
    kml_path = f"{out_prefix}_review.kml"
    write_review_kml(waypoints, sortie_index, conv, kml_path)
    files.append(kml_path)

    for s in np.unique(sortie_index):
        sel = np.where(sortie_index == s)[0]
        kmz = f"{out_prefix}_sortie{int(s) + 1}.kmz"
        write_wpml_kmz([waypoints[i] for i in sel], conv, kmz,
                       speed_ms=speed_ms, takeoff_z_m=takeoff_z_m)
        files.append(kmz)
    return files, rt_err
