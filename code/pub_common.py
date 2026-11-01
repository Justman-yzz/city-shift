"""pub_common.py — 공개용 R1 파이프라인 공용 함수 (부산교통공사 원본·공휴일 목록만 사용).

노선 그래프, 사건 검출, 앵커 클러스터링, 요약 함수와 단일 기준선 SQL(모든 역에 같은 규칙, 조건부 분할 없음).
외부 일정 테이블을 참조하지 않는다.
"""
import collections, pathlib, duckdb, numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
PUB = ROOT / "public"
INTERNAL = ROOT / "out" / "public_internal"
BUILD_DB = ROOT / "data" / "public_build.duckdb"
SOURCE_DB = ROOT / "data" / "cityshift.duckdb"
# 공개 파이프라인이 원본 DB에서 가져오는 테이블(전부 부산교통공사 원본·공휴일 목록에서 생성)
SOURCE_TABLES = ["raw_wide", "hourly", "hourly_flag", "special_days"]


def build_db():
    """원본 DB에서 4개 테이블만 새 DB로 복사한다 → 공개 계산 DB에는 이 4개 외의 테이블이 존재하지 않는다."""
    if BUILD_DB.exists(): BUILD_DB.unlink()
    db = duckdb.connect(str(BUILD_DB))
    db.execute(f"ATTACH '{SOURCE_DB}' AS src (READ_ONLY)")
    for t in SOURCE_TABLES:
        db.execute(f"CREATE TABLE {t} AS SELECT * FROM src.{t}")
    db.execute("DETACH src")
    return db


def connect():
    return duckdb.connect(str(BUILD_DB))


# ---------------- 노선 그래프 ----------------
def build_graph(db):
    sta = db.execute("SELECT DISTINCT sta_cd, sta_nm FROM raw_wide ORDER BY 1").df()
    name = dict(zip(sta.sta_cd, sta.sta_nm))
    line = lambda c: 1 if c < 200 else c // 100
    G = collections.defaultdict(set); by_line = collections.defaultdict(list)
    for c in sorted(name): by_line[line(c)].append(c)
    for cs in by_line.values():
        for a, b in zip(cs, cs[1:]): G[a].add(b); G[b].add(a)
    by_name = collections.defaultdict(list)
    for c, n in name.items(): by_name[n].append(c)
    for cs in by_name.values():
        for a in cs:
            for b in cs:
                if a != b: G[a].add(b)
    inv = {n: c for c, n in name.items()}
    for first, hub in [(by_line[3][0], "수영"), (by_line[4][0], "미남")]:
        G[first].add(inv[hub]); G[inv[hub]].add(first)
    # 전체 최단거리(BFS)
    dist = {}
    for s in name:
        d = {s: 0}; q = collections.deque([s])
        while q:
            u = q.popleft()
            for v in G[u]:
                if v not in d: d[v] = d[u] + 1; q.append(v)
        dist[s] = d
    return name, dist


def area(db, names, hops):
    name, dist = build_graph(db)
    inv = collections.defaultdict(list)
    for c, n in name.items(): inv[n].append(c)
    seeds = [c for n in names for c in inv[n]]
    return sorted({c for s in seeds for c, d in dist[s].items() if d <= hops})



# ---------------- 기준선 SQL (단일 규칙) ----------------
def baseline_sql(table, window=56, exclude_special=True):
    """같은 역·구분·시간대·요일, ±window일, 자기 자신 제외 중앙값. 특수일은 표본에서 제외(옵션)."""
    filt_expr = "FILTER (WHERE NOT is_special)" if exclude_special else ""
    return f"""
CREATE OR REPLACE TABLE {table} AS
WITH s AS (
  SELECT *,
    count(cnt)              {filt_expr} OVER w AS n_base,
    median(cnt)             {filt_expr} OVER w AS base_med,
    quantile_cont(cnt,0.25) {filt_expr} OVER w AS base_q1,
    quantile_cont(cnt,0.75) {filt_expr} OVER w AS base_q3
  FROM hourly_flag
  WINDOW w AS (PARTITION BY sta_cd, gubn, hour, dow_n ORDER BY ymd
               RANGE BETWEEN INTERVAL {window} DAYS PRECEDING AND INTERVAL {window} DAYS FOLLOWING
               EXCLUDE CURRENT ROW)
)
SELECT *,
  cnt - base_med AS delta,
  (cnt + 10.0) / (base_med + 10.0) AS ratio,
  (cnt - base_med) / greatest((base_q3 - base_q1) / 1.349, sqrt(base_med), 3.0) AS z,
  n_base < 8 AS low_sample
FROM s;
"""


# ---------------- 사건 검출 (1차와 동일 규칙, 임계값만 인자) ----------------
def station_events(db, scored, z=4.0, delta=150, ratio=1.5):
    lo = 1 / ratio
    return db.execute(f"""
WITH a AS (
  SELECT *, CASE WHEN delta > 0 THEN 'surge' ELSE 'drop' END AS direction FROM {scored}
  WHERE (z >= {z} AND delta >= {delta} AND ratio >= {ratio}) OR (z <= -{z} AND delta <= -{delta} AND ratio <= {lo})
), hrs AS (SELECT DISTINCT sta_cd, ymd, direction, hour FROM a),
isl AS (SELECT *, hour - lag(hour) OVER (PARTITION BY sta_cd, ymd, direction ORDER BY hour) gap FROM hrs),
grp AS (SELECT *, sum(CASE WHEN gap IS NULL OR gap > 2 THEN 1 ELSE 0 END) OVER (PARTITION BY sta_cd, ymd, direction ORDER BY hour) seg FROM isl),
seg AS (SELECT sta_cd, ymd, direction, seg, min(hour) h_start, max(hour) h_end FROM grp GROUP BY ALL)
SELECT s.sta_cd, any_value(a.sta_nm) sta_nm, s.ymd, any_value(a.dow) dow, s.direction, s.h_start, s.h_end,
       count(*) n_cells, sum(a.cnt) obs, sum(a.base_med) base, sum(a.delta) excess,
       coalesce(sum(a.delta) FILTER (WHERE a.gubn='승차'),0) excess_board,
       coalesce(sum(a.delta) FILTER (WHERE a.gubn='하차'),0) excess_alight,
       max(abs(a.z)) max_abs_z, min(a.n_base) min_n_base, bool_or(a.low_sample) low_sample,
       bool_or(a.is_special) is_special, any_value(a.special_name) special_name,
       abs(sum(a.delta)) * abs(log2((sum(a.cnt)+10.0)/(sum(a.base_med)+10.0))) score
FROM seg s JOIN a ON a.sta_cd=s.sta_cd AND a.ymd=s.ymd AND a.direction=s.direction AND a.hour BETWEEN s.h_start AND s.h_end
GROUP BY s.sta_cd, s.ymd, s.direction, s.seg, s.h_start, s.h_end
""").df()


# ---------------- 클러스터링 ----------------
def cluster(ev, dist, mode="anchor", hops=3):
    """mode='chain'  : 1차 방식(연결요소, 이웃의 이웃도 연결 → 연쇄 가능)
       mode='anchor' : 2차 방식. 점수가 가장 큰 역 사건을 앵커로 두고, 앵커에서 hops 이내·시간 겹침(±1h)인 사건만 흡수.
                       흡수 판정은 '앵커 기준'이라 연쇄가 생기지 않는다."""
    ev = ev.sort_values("score", ascending=False).reset_index(drop=True)
    ev["cid"] = -1; cid = 0
    for (ymd, direc), grp in ev.groupby(["ymd", "direction"], sort=False):
        idx = list(grp.sort_values("score", ascending=False).index)
        if mode == "chain":
            parent = {i: i for i in idx}
            def find(i):
                while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
                return i
            for x in idx:
                for y in idx:
                    if y <= x: continue
                    a, b = ev.loc[x], ev.loc[y]
                    if a.h_start <= b.h_end + 1 and b.h_start <= a.h_end + 1 and dist[a.sta_cd].get(b.sta_cd, 99) <= hops:
                        parent[find(y)] = find(x)
            roots = {}
            for i in idx:
                r = find(i)
                if r not in roots: roots[r] = cid; cid += 1
                ev.loc[i, "cid"] = roots[r]
        else:
            free = list(idx)
            while free:
                a = free.pop(0); A = ev.loc[a]; ev.loc[a, "cid"] = cid; keep = []
                for b in free:
                    B = ev.loc[b]
                    if dist[A.sta_cd].get(B.sta_cd, 99) <= hops and A.h_start <= B.h_end + 1 and B.h_start <= A.h_end + 1:
                        ev.loc[b, "cid"] = cid
                    else:
                        keep.append(b)
                free = keep; cid += 1
    return ev


def summarize(ev, dist):
    rows = []
    for c, g in ev.groupby("cid"):
        g = g.sort_values("score", ascending=False); a = g.iloc[0]
        codes = list(g.sta_cd)
        diam = max(dist[x].get(y, 99) for x in codes for y in codes)
        rows.append(dict(cid=c, ymd=a.ymd, dow=a.dow, direction=a.direction, anchor=a.sta_nm, anchor_cd=int(a.sta_cd),
                         n_stations=g.sta_cd.nunique(), diameter_hops=diam,
                         stations="; ".join(f"{r.sta_nm}({int(r.h_start)}-{int(r.h_end)+1}시)" for r in g.itertuples()),
                         h_start=int(g.h_start.min()), h_end_excl=int(g.h_end.max()) + 1,
                         obs=int(g.obs.sum()), baseline=int(g.base.sum()), excess=int(g.excess.sum()),
                         excess_board=int(g.excess_board.sum()), excess_alight=int(g.excess_alight.sum()),
                         pct_change=round(100 * (g.obs.sum() / max(g.base.sum(), 1) - 1), 1),
                         max_abs_z=round(float(g.max_abs_z.max()), 1), min_n_base=int(g.min_n_base.min()),
                         low_sample=bool(g.low_sample.any()), is_special=bool(g.is_special.any()),
                         special_name=(g.special_name.dropna().iloc[0] if g.is_special.any() else ""),
                         score=float(g.score.sum())))
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


