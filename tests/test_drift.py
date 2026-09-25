"""单点漂移（跳点）检测与修复的单元测试。

可直接运行：python3 tests/test_drift.py
或使用 pytest：python3 -m pytest tests/
"""
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.drift import MAX_SPEED_MPS, MIN_OFFSET_M, fix_drift
from app.geo import EARTH_RADIUS_M, TrackPoint, haversine_m, point_segment_distance_m
from app.gpxutil import build_gpx, parse_gpx

T0 = datetime(2026, 5, 1, 8, 0, 0, tzinfo=timezone.utc)

# 纬度 31° 处 0.01° 经度差的水平距离（约 954 m），用作跳点偏移量
JUMP_LON = 0.01


def make_seg(points):
    """points: [(lat, lon, 秒偏移), ...]"""
    return [
        TrackPoint(lat=la, lon=lo, ele=500.0, time=T0 + timedelta(seconds=s))
        for la, lo, s in points
    ]


def walking_seg(n_points):
    """沿正北等距（0.0001° 纬度 ≈ 11.1 m）等速（10 s 一步 ≈ 1.1 m/s）的轨迹段。"""
    return make_seg([
        (31.0 + i * 0.0001, 121.0, i * 10.0) for i in range(n_points)
    ])


def lat_offset(meters):
    """恰好产生指定球面距离的纬度差（度）。"""
    return math.degrees(meters / EARTH_RADIUS_M)


def test_single_jump_removed():
    """直线徒步轨迹中夹一个跳点：命中并删除，首尾点保留，里程下降。"""
    seg = walking_seg(10)
    jumper = TrackPoint(lat=31.0003, lon=121.0 + JUMP_LON, ele=500.0,
                        time=T0 + timedelta(seconds=30))
    seg[3] = jumper

    res = fix_drift([seg])
    assert len(res.hits) == 1
    hit = res.hits[0]
    assert hit.seg_index == 1
    assert hit.point_index == 4      # 原始序号（从 1 开始）
    assert hit.round == 1
    assert hit.point is jumper
    assert hit.speed_in_mps > MAX_SPEED_MPS
    assert hit.speed_out_mps > MAX_SPEED_MPS
    assert hit.speed_direct_mps <= MAX_SPEED_MPS
    assert hit.offset_m > MIN_OFFSET_M

    fixed = res.fixed_segments[0]
    assert len(fixed) == 9
    assert fixed[0] is seg[0] and fixed[-1] is seg[-1]  # 首尾点保留
    assert all(p is not jumper for p in fixed)
    assert len(seg) == 10            # 原始轨迹不被修改（遮蔽/配速仍用它）

    assert res.distance_before_m > 1800.0   # 往返跳点约 2×954 m
    assert res.distance_after_m < 150.0     # 修复后回到约 100 m 的徒步里程
    assert res.distance_after_m < res.distance_before_m

    # 修复结果可正常导出并解析
    segments, name = parse_gpx(build_gpx(res.fixed_segments, "t-fixed"))
    assert name == "t-fixed"
    assert len(segments) == 1 and len(segments[0]) == 9


def test_clean_track_untouched():
    """正常轨迹：无命中，逐点保持一致，里程不变。"""
    seg = walking_seg(12)
    res = fix_drift([seg])
    assert res.hits == []
    assert res.fixed_segments[0] == seg
    assert abs(res.distance_before_m - res.distance_after_m) < 1e-6


def test_slow_detour_not_flagged():
    """偏移较远但速度不超过 12 m/s（时间足够长）：不是跳点。"""
    seg = walking_seg(6)
    # 偏移约 95 m（0.001° 经度），10 s 间隔 → 约 9.5 m/s
    seg[2] = TrackPoint(lat=31.0002, lon=121.001, ele=500.0,
                        time=T0 + timedelta(seconds=20))
    res = fix_drift([seg])
    assert res.hits == []
    assert res.fixed_segments[0] == seg


def test_fast_direct_not_flagged():
    """进出都快、绕过该点的首尾速度也超过 12 m/s：是真实高速移动，不是跳点。"""
    seg = make_seg([
        (31.0, 121.0, 0),
        (31.001, 121.002, 5),   # 偏离航线约 111 m，但全程约 38 m/s
        (31.0, 121.004, 10),
    ])
    res = fix_drift([seg])
    assert res.hits == []


def test_small_offset_not_flagged():
    """速度条件满足但偏离不超过 40 m：不是跳点。"""
    seg = make_seg([
        (31.0, 121.0, 0),
        (31.0003, 121.0, 1),   # 约 33 m 偏离，1 s → 约 33 m/s
        (31.0, 121.0, 2),
    ])
    res = fix_drift([seg])
    assert res.hits == []


def test_multiple_jumps_removed_one_per_round():
    """同一段内两个跳点：逐轮各删一个，按原始顺序先删靠前者。"""
    seg = walking_seg(12)
    seg[3] = TrackPoint(lat=31.0003, lon=121.0 + JUMP_LON, ele=500.0,
                        time=T0 + timedelta(seconds=30))
    seg[8] = TrackPoint(lat=31.0008, lon=121.0 + JUMP_LON, ele=500.0,
                        time=T0 + timedelta(seconds=80))

    res = fix_drift([seg])
    assert len(res.hits) == 2
    assert res.hits[0].point_index == 4 and res.hits[0].round == 1
    assert res.hits[1].point_index == 9 and res.hits[1].round == 2
    assert res.hits[0].point_index < res.hits[1].point_index
    assert len(res.fixed_segments[0]) == 10
    assert res.fixed_segments[0][0] is seg[0]
    assert res.fixed_segments[0][-1] is seg[-1]


def test_adjacent_outliers_not_flagged():
    """相邻两个异常点：绕过任一个的首尾速度仍超过 12 m/s，均不判为跳点。"""
    seg = make_seg([
        (31.0, 121.0, 0),
        (31.0, 121.0 + JUMP_LON, 10),            # 远点 1
        (31.0, 121.0 + JUMP_LON * 1.1, 20),      # 远点 2（与远点 1 相邻）
        (31.0, 121.0, 30),
    ])
    res = fix_drift([seg])
    assert res.hits == []
    assert len(res.fixed_segments[0]) == 4


def test_endpoints_preserved():
    """首点或末点再异常也不会被删：跳点判定只针对中间点。"""
    seg1 = walking_seg(6)
    seg1[0] = TrackPoint(lat=31.0, lon=121.05, ele=500.0, time=T0)  # 首点是远点
    seg2 = walking_seg(6)
    seg2[-1] = TrackPoint(lat=31.0005, lon=121.05, ele=500.0,
                          time=T0 + timedelta(seconds=50))          # 末点是远点

    res = fix_drift([seg1, seg2])
    assert res.hits == []
    assert len(res.fixed_segments[0]) == 6
    assert len(res.fixed_segments[1]) == 6
    assert res.fixed_segments[0][0] is seg1[0]
    assert res.fixed_segments[1][-1] is seg2[-1]


def test_non_increasing_time_rejected():
    """零时差或倒序：报明确错误，不修复。"""
    ok = walking_seg(3)
    fix_drift([ok])  # 严格递增不报错

    same_time = make_seg([(31.0, 121.0, 0), (31.001, 121.0, 0)])
    backwards = make_seg([(31.0, 121.0, 10), (31.001, 121.0, 5)])
    for bad in (same_time, backwards):
        try:
            fix_drift([bad])
        except ValueError as exc:
            assert "未严格递增" in str(exc) and "漂移修复" in str(exc)
        else:
            raise AssertionError("时间不严格递增的轨迹应被拒绝")

    # 错误信息指明是第几个 trkseg
    try:
        fix_drift([walking_seg(2), backwards])
    except ValueError as exc:
        assert "第 2 个 trkseg" in str(exc)
    else:
        raise AssertionError("第二个 trkseg 时间倒序应被拒绝")


def test_no_cross_trkseg_and_short_segments():
    """不跨 trkseg 判断；不足 3 个点的段没有中间点，不可能命中。"""
    seg1 = walking_seg(3)                                   # 末尾在 (31.0002, 121.0)
    seg2 = make_seg([(31.05, 121.0, 30), (31.0501, 121.0, 40),
                     (31.0502, 121.0, 50)])                 # 与 seg1 末尾相距数公里
    seg3 = make_seg([(31.0, 121.0, 60), (31.0, 121.05, 70)])  # 两点段，跨度巨大

    res = fix_drift([seg1, seg2, seg3])
    assert res.hits == []
    assert [len(s) for s in res.fixed_segments] == [3, 3, 2]
    assert abs(res.distance_before_m - res.distance_after_m) < 1e-6


def test_speed_threshold():
    """进出速度刚好不超过 / 超过 12 m/s 的边界行为。"""
    # 偏移约 100 m、10 s → 约 10 m/s ≤ 12：不命中（其余条件都满足）
    slow = make_seg([
        (31.0, 121.0, 0),
        (31.0 + lat_offset(100.0), 121.0, 10),
        (31.0, 121.0, 20),
    ])
    assert fix_drift([slow]).hits == []

    # 同样偏移、8 s → 约 12.5 m/s > 12：命中
    fast = make_seg([
        (31.0, 121.0, 0),
        (31.0 + lat_offset(100.0), 121.0, 8),
        (31.0, 121.0, 16),
    ])
    res = fix_drift([fast])
    assert len(res.hits) == 1
    assert res.hits[0].speed_in_mps > MAX_SPEED_MPS


def test_offset_threshold():
    """偏离刚好不超过 / 超过 40 m 的边界行为（速度条件均满足）。"""
    near = make_seg([
        (31.0, 121.0, 0),
        (31.0 + lat_offset(39.9), 121.0, 2),   # 约 20 m/s，偏离 39.9 m
        (31.0, 121.0, 4),
    ])
    assert fix_drift([near]).hits == []

    far = make_seg([
        (31.0, 121.0, 0),
        (31.0 + lat_offset(40.1), 121.0, 2),   # 偏离 40.1 m
        (31.0, 121.0, 4),
    ])
    res = fix_drift([far])
    assert len(res.hits) == 1
    assert res.hits[0].offset_m > MIN_OFFSET_M


def test_point_segment_distance_m():
    """点到线段水平最短距离：垂足、端点收敛、零长线段。"""
    a = TrackPoint(lat=31.0, lon=121.0)
    b = TrackPoint(lat=31.0, lon=121.001)
    mid_foot = TrackPoint(lat=31.0, lon=121.0005)

    p = TrackPoint(lat=31.0005, lon=121.0005)   # 线段中点正北约 55.6 m
    d = point_segment_distance_m(p, a, b)
    assert abs(d - haversine_m(p, mid_foot)) < 0.5

    beyond = TrackPoint(lat=31.0005, lon=120.999)  # 垂足在线段外 → 收敛到端点 a
    assert abs(point_segment_distance_m(beyond, a, b)
               - haversine_m(beyond, a)) < 0.5

    # 零长线段（a == b）退化为点到点的距离（局部投影与球面距离误差在毫米级）
    assert abs(point_segment_distance_m(p, a, a) - haversine_m(p, a)) < 1e-3


def _run_all():
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
