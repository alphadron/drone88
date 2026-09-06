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
WP_COLOR = "ff00ffff"        # 일반 웨이포인트 아이콘(노랑, 라인과 대비되도록)
WP_START_COLOR = "ff00ff00"  # 시작점(녹색)
WP_END_COLOR = "ff0000ff"    # 종료점(적색)
SURFACE_FILL_COLOR = "5f1478d2"    # 기준면 채움(반투명 주황빛 갈색, aabbggrr)
SURFACE_LINE_COLOR = "ff1478d2"    # 기준면 외곽선
SURFACE_MAX_FACES = 5000           # 이보다 큰 메시는 KML 비대화 방지 위해 생략
ANCHOR_ICON = "http://maps.google.com/mapfiles/kml/pushpin/wht-pushpin.png"
BOUNDARY_LINE_COLOR = "ff0000ff"    # 촬영 경계(사용자 지정) 외곽선 — 적색, aabbggrr


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
_ALT_MODES = {"absolute": simplekml.AltitudeMode.absolute,
              "relativeToGround": simplekml.AltitudeMode.relativetoground}


def _add_surface_polygons(kml, conv: EnuConverter, mesh, max_faces: int = None,
                          altmode=simplekml.AltitudeMode.absolute):
    """기준면 메시를 반투명 폴리곤으로 KML에 추가(면수 과다 시 생략)."""
    max_faces = SURFACE_MAX_FACES if max_faces is None else max_faces
    if mesh is None or len(mesh.faces) > max_faces:
        return
    verts_llh = conv.to_wgs84(np.asarray(mesh.vertices, float))
    sfol = kml.newfolder(name="대상 기준면")
    sfol.visibility = 1
    for tri in mesh.faces:
        ring = [tuple(verts_llh[i]) for i in tri] + [tuple(verts_llh[tri[0]])]
        pol = sfol.newpolygon(outerboundaryis=ring)
        pol.altitudemode = altmode
        pol.style.polystyle.color = SURFACE_FILL_COLOR
        pol.style.polystyle.fill = 1
        pol.style.polystyle.outline = 1
        pol.style.linestyle.color = SURFACE_LINE_COLOR
        pol.style.linestyle.width = 1


def _add_anchor_marker(kml, conv: EnuConverter):
    """ENU 원점(기준점)을 지면에 고정된 핀으로 표시 — 기준점이 실제 현장과
    다른 곳(엉뚱한 실측 위치)에 잡혔는지 Earth 화면에서 바로 확인하기 위함."""
    lon, lat, h = conv.to_wgs84(np.zeros((1, 3)))[0]
    p = kml.newpoint(name="ENU 원점(기준점) — 위치 확인",
                     coords=[(lon, lat, 0.0)])
    p.altitudemode = simplekml.AltitudeMode.clamptoground
    p.style.iconstyle.icon.href = ANCHOR_ICON
    p.style.iconstyle.scale = 1.2
    p.style.labelstyle.scale = 1.0
    p.description = (f"anchor lat={conv.anchor.lat:.7f}, lon={conv.anchor.lon:.7f}, "
                     f"h={conv.anchor.h:.2f} m — 이 지점이 실제 시설물(사면 등) "
                     f"위치와 다르면 --anchor/--epsg 값을 확인하십시오.")


def _add_boundary_outline(kml, conv: EnuConverter, boundary_en):
    """사용자가 Google Earth(Pro)에서 그려 지정한 촬영 경계 다각형을
    지면에 고정된 굵은 적색 외곽선으로 표시(실제 웨이포인트 제한 범위와
    화면상 경계가 일치하는지 육안 확인용)."""
    if boundary_en is None:
        return
    ring = np.vstack([boundary_en, boundary_en[:1]])
    llh = conv.to_wgs84(np.column_stack([ring, np.zeros(len(ring))]))
    fol = kml.newfolder(name="촬영 경계(사용자 지정)")
    ls = fol.newlinestring(name="경계", coords=[tuple(p) for p in llh])
    ls.altitudemode = simplekml.AltitudeMode.clamptoground
    ls.tessellate = 1
    ls.style.linestyle.color = BOUNDARY_LINE_COLOR
    ls.style.linestyle.width = 4


def write_review_kml(waypoints, sortie_index, conv: EnuConverter, path: str,
                     name: str = "FacilityPath Mission", wp_label_step: int = 5,
                     surface_mesh=None, surface_max_faces=None,
                     altitude_mode: str = "absolute", wp_extrude: bool = True,
                     show_anchor_marker: bool = True, boundary_en=None):
    """
    소티별 경로 라인 + 전 웨이포인트 포인트(짐벌·헤딩 ExtendedData) + (선택)
    기준면 반투명 폴리곤 + ENU 원점 핀을 담은 Google Earth 검토용 KML.
    altitude_mode: "absolute"(기준점 타원체고 + z — Earth 실제 지형/기준면
    대비 이격 확인용, 기준점 고도가 부정확하면 경로가 지형에 파묻혀 안 보일
    수 있음) 또는 "relativeToGround"(실제 지형 표면 기준 상대고도 — 기준점
    고도 오차와 무관하게 항상 지표 위에 보이므로 화면 확인용으로 더 안전).
    점은 전부 아이콘으로 표시하고, wp_label_step=1이면 전 웨이포인트 이름을
    모두 표시한다(기본 5면 과밀 방지를 위해 라벨만 축소, 시작·끝점은 항상
    라벨 표시 + 색상 구분).
    wp_extrude: 각 점에서 지면까지 수직 안내선을 그릴지 여부. 기준점 고도가
    실제 지형과 크게 어긋난 상태에서는 이 안내선이 비정상적으로 길게 그려져
    화면을 뒤덮을 수 있으므로(§실측 사례), 그런 경우 False로 끈다.
    show_anchor_marker: ENU 원점을 지면 고정 핀으로 표시할지 여부(기본 표시
    — 기준점 위치가 실제 현장과 다른지 화면에서 바로 확인 가능).
    """
    altmode = _ALT_MODES[altitude_mode]
    enu = np.array([w.position for w in waypoints])
    llh = conv.to_wgs84(enu)
    kml = simplekml.Kml(name=name)
    kml.document.open = 1

    if show_anchor_marker:
        _add_anchor_marker(kml, conv)

    _add_surface_polygons(kml, conv, surface_mesh, max_faces=surface_max_faces,
                          altmode=altmode)
    _add_boundary_outline(kml, conv, boundary_en)

    n = len(waypoints)
    for s in np.unique(sortie_index):
        sel = np.where(sortie_index == s)[0]
        color = SORTIE_COLORS[int(s) % len(SORTIE_COLORS)]
        fol = kml.newfolder(name=f"Sortie {s + 1}")
        ls = fol.newlinestring(name=f"Path S{s + 1}",
                               coords=[tuple(llh[i]) for i in sel])
        ls.altitudemode = altmode
        ls.tessellate = 1
        ls.style.linestyle.color = color
        ls.style.linestyle.width = 6

        pfol = fol.newfolder(name="Waypoints")
        for k, i in enumerate(sel):
            w = waypoints[i]
            is_end = (i == 0 or i == n - 1)
            show_label = is_end or k % wp_label_step == 0 or k == len(sel) - 1
            p = pfol.newpoint(name=f"WP{i}", coords=[tuple(llh[i])])
            p.altitudemode = altmode
            p.extrude = 1 if wp_extrude else 0              # 지면까지 수직 안내선(선택)
            p.style.iconstyle.icon.href = (
                "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png")
            p.style.iconstyle.scale = 1.1 if is_end else (0.85 if show_label else 0.65)
            p.style.iconstyle.color = (WP_START_COLOR if i == 0 else
                                        WP_END_COLOR if i == n - 1 else WP_COLOR)
            p.style.labelstyle.scale = 0.75 if show_label else 0.0
            p.style.labelstyle.color = "ffffffff"
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
                   takeoff_z_m: float = 0.0, surface_mesh=None,
                   surface_max_faces=None, wp_label_step: int = 5,
                   altitude_mode: str = "absolute", mission_name: str = "FacilityPath Mission",
                   wp_extrude: bool = True, show_anchor_marker: bool = True,
                   boundary_en=None):
    """
    통합 내보내기:
      {prefix}_review.kml            — 전 소티 Earth 검토용(+ 기준면 폴리곤 + 원점 핀)
      {prefix}_sortie{N}.kmz         — 소티별 Pilot 2 WPML
    surface_mesh: 대상 기준면(로컬 ENU Trimesh, 선택) — 지정 시 review KML에
    반투명 폴리곤으로 함께 표시. surface_max_faces 초과 시(기본
    SURFACE_MAX_FACES) 자동 생략해 KML 비대화를 막는다.
    wp_label_step: 웨이포인트 이름 라벨 표시 간격(1=전부 표시).
    altitude_mode/wp_extrude/show_anchor_marker: write_review_kml 참고.
    mission_name: review KML의 Document 이름 — 여러 결과물을 Google Earth
    Pro에 동시에 불러왔을 때 Places 패널에서 구분할 수 있도록 실행마다
    다르게(입력 파일명 등 반영) 지정하는 것을 권장한다.
    boundary_en: 사용자가 Google Earth(Pro)에서 그린 촬영 경계 다각형
    (로컬 ENU E,N, 선택) — 지정 시 review KML에 굵은 적색 외곽선으로
    함께 표시(웨이포인트 자체는 planner_base.clip_to_boundary로 이미
    제한된 상태로 들어온다).
    반환: (생성 파일 목록, ENU 왕복오차[m])
    """
    conv = EnuConverter(anchor)
    enu = np.array([w.position for w in waypoints])
    rt_err = conv.roundtrip_error_m(enu)

    files = []
    kml_path = f"{out_prefix}_review.kml"
    write_review_kml(waypoints, sortie_index, conv, kml_path, name=mission_name,
                     surface_mesh=surface_mesh, surface_max_faces=surface_max_faces,
                     wp_label_step=wp_label_step, altitude_mode=altitude_mode,
                     wp_extrude=wp_extrude, show_anchor_marker=show_anchor_marker,
                     boundary_en=boundary_en)
    files.append(kml_path)

    for s in np.unique(sortie_index):
        sel = np.where(sortie_index == s)[0]
        kmz = f"{out_prefix}_sortie{int(s) + 1}.kmz"
        write_wpml_kmz([waypoints[i] for i in sel], conv, kmz,
                       speed_ms=speed_ms, takeoff_z_m=takeoff_z_m)
        files.append(kmz)
    return files, rt_err
