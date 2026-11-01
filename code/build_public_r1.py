"""build_public_r1.py — CITY//SHIFT 공개용 R1 데이터 생성 (원본 승하차 + 공휴일 목록 + 공개 출처 사실만 사용)

실행(프로젝트 루트): python3 src/public/build_public_r1.py
입력: data/cityshift.duckdb 의 raw_wide·hourly·hourly_flag·special_days (부산교통공사 원본과 공휴일 목록에서 생성)
      public/data/catalog/facts.csv (공개 출처 사실 카탈로그), src/public/evidence_map_public.csv (사건-사실 매핑)
산출: public/data/a/*.csv, public/data/b/*.csv, out/public_internal/*.csv(공개 제외 목록)

규칙 요약
 A 기준선 : 같은 역·구분·시간대·요일, ±56일 중앙값(자기 제외), 특수일은 표본에서 제외. 모든 역에 같은 규칙.
 A 사건   : z≥4·증감≥150·배율≥1.5(감소는 대칭) → 역 사건 → 앵커 3정거장 클러스터 → 같은 시각 20개 역 이상이면 도시 전역 사건으로 묶음
 C안      : 앵커가 사직권 7개 역(사직·종합운동장에서 2정거장 이내)인 '지역' 사건은 공개 목록에서 제외(검증 전). 도시 전역 사건은 유지
 분석 일관성 : 기준 변형 7개(기간 28·84일, 특수일 포함, z 3·5, 증감 300, 배율 2) 중 같은 사건이 다시 잡힌 수
 B 기준선 : 관련 역 합계, 같은 요일 ±56일 '행사가 없던 평소 날'(특수일·카탈로그 행사일 제외) 중앙값
 B 판정   : 평소 날끼리 비교한 증감률 분포(평일/주말 구분)의 95백분위 초과 → 늘었음, 5백분위 미만 → 줄었음. 표본 1~7일 → 판정 보류, 0일 → 불가
"""
import json, numpy as np, pandas as pd
from pub_common import (build_db, build_graph, area, baseline_sql, station_events, cluster, summarize, PUB, INTERNAL, ROOT)

A_DIR = PUB / "data" / "a"; B_DIR = PUB / "data" / "b"; C_DIR = PUB / "data" / "catalog"
for d in (A_DIR, B_DIR, C_DIR, INTERNAL): d.mkdir(parents=True, exist_ok=True)
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)

db = build_db()
assert not db.execute("SELECT count(*) FROM information_schema.tables WHERE table_name NOT IN ('raw_wide','hourly','hourly_flag','special_days')").fetchone()[0]
name, dist = build_graph(db)
inv = {}
for c, n in name.items(): inv.setdefault(n, []).append(c)
codes = lambda names: [c for n in names for c in inv[n]]
STADIUM = set(area(db, ["사직", "종합운동장"], 2))
assert sorted(name[c] for c in STADIUM) == ["거제", "동래", "만덕", "미남", "사직", "연산", "종합운동장"]
log("공개 계산 DB 테이블:", [r[0] for r in db.execute("SELECT table_name FROM information_schema.tables ORDER BY 1").fetchall()])

# ---------------- 1. 기준선 ----------------
for w, ex in [(56, True), (28, True), (84, True), (56, False)]:
    db.execute(baseline_sql(f"pb_{w}_{int(ex)}", w, ex))
VARIANTS = {"MAIN": (56, True, 4, 150, 1.5),
            "S_win28": (28, True, 4, 150, 1.5), "S_win84": (84, True, 4, 150, 1.5), "S_nospecial": (56, False, 4, 150, 1.5),
            "S_z3": (56, True, 3, 150, 1.5), "S_z5": (56, True, 5, 150, 1.5), "S_delta300": (56, True, 4, 300, 1.5), "S_ratio2": (56, True, 4, 150, 2.0)}
tbl = lambda v: f"pb_{VARIANTS[v][0]}_{int(VARIANTS[v][1])}"

def city_hours(v, direction):
    w, ex, z, d, r = VARIANTS[v]
    cond = f"z>={z} AND delta>={d} AND ratio>={r}" if direction == "surge" else f"z<=-{z} AND delta<=-{d} AND ratio<={1/r}"
    df = db.execute(f"SELECT ymd, hour, count(DISTINCT sta_cd) n FROM {tbl(v)} WHERE {cond} GROUP BY ALL").df()
    df["ymd"] = pd.to_datetime(df.ymd); df["direction"] = direction
    return df

# ---------------- 2. 사건 (주 기준) ----------------
CITY_MIN, WIN_MIN, PAD = 20, 8, 2
ev = cluster(station_events(db, tbl("MAIN"), 4, 150, 1.5), dist, "anchor")
cl = summarize(ev, dist); cl["ymd"] = pd.to_datetime(cl.ymd); ev["ymd"] = pd.to_datetime(ev.ymd)
hc = pd.concat([city_hours("MAIN", "surge"), city_hours("MAIN", "drop")])
city = []
for (d, dr), g in hc.groupby(["ymd", "direction"]):
    if g.n.max() < CITY_MIN: continue
    ok = set(g[g.n >= WIN_MIN].hour); pk = int(g.loc[g.n.idxmax(), "hour"]); lo = hi = pk
    while lo - 1 in ok: lo -= 1
    while hi + 1 in ok: hi += 1
    city.append(dict(ymd=d, direction=dr, peak_n_stations=int(g.n.max()), peak_hour=pk, win_start=max(lo - PAD, 1), win_end=min(hi + PAD, 24)))
city = pd.DataFrame(city)
cl["event_id"] = ["L%05d" % c for c in cl.cid]; cl["scale"] = "지역"
for r in city.itertuples():
    m = (cl.ymd == r.ymd) & (cl.direction == r.direction) & (cl.h_start <= r.win_end) & (cl.h_end_excl - 1 >= r.win_start)
    cl.loc[m, "event_id"] = f"C{r.ymd:%Y%m%d}{r.direction[0]}"; cl.loc[m, "scale"] = "도시 전역"

def agg(g):
    g = g.sort_values("score", ascending=False); a = g.iloc[0]
    return pd.Series(dict(ymd=a.ymd, dow=a.dow, direction=a.direction, scale=a.scale, anchor=a.anchor, anchor_cd=a.anchor_cd,
        n_local_clusters=len(g), n_stations=int(g.n_stations.sum()), core_share=round(a.score / g.score.sum(), 3),
        stations=" || ".join(g.stations.head(6)) + (" || …" if len(g) > 6 else ""),
        h_start=int(g.h_start.min()), h_end_excl=int(g.h_end_excl.max()),
        obs=int(g.obs.sum()), baseline=int(g.baseline.sum()), excess=int(g.excess.sum()),
        excess_alight=int(g.excess_alight.sum()), excess_board=int(g.excess_board.sum()),
        pct_change=round(100 * (g.obs.sum() / max(g.baseline.sum(), 1) - 1), 1),
        max_abs_z=round(float(g.max_abs_z.max()), 1), min_n_base=int(g.min_n_base.min()), low_sample=bool(g.low_sample.any()),
        is_special=bool(g.is_special.any()), special_name=(g.special_name.dropna().iloc[0] if g.special_name.astype(bool).any() else ""),
        score=float(g.score.sum()), member_cids=",".join(str(x) for x in g.cid)))
E = cl.groupby("event_id").apply(agg, include_groups=False).reset_index().sort_values("score", ascending=False).reset_index(drop=True)
E["stadium_local"] = E.anchor_cd.isin(STADIUM) & (E.scale == "지역")
CATS = {"main": ("surge", False, 50), "calendar": ("surge", True, 20), "drops": ("drop", False, 20)}
def pick(df, cat, n=None):
    d, sp, k = CATS[cat]
    x = df[(df.direction == d) & (df.is_special == sp)]
    return x.head(n or k).reset_index(drop=True)
LISTS = {c: pick(E[~E.stadium_local], c) for c in CATS}
SEPARATED = E[E.stadium_local & ((E.direction == "surge") | ~E.is_special)].reset_index(drop=True)

# ---------------- 3. 분석 일관성 (변형 7개) ----------------
VLIST, CITYMAX = {}, {}
for v in VARIANTS:
    if v == "MAIN": continue
    w, ex, z, d, r = VARIANTS[v]
    c = summarize(cluster(station_events(db, tbl(v), z, d, r), dist, "anchor"), dist)
    c["ymd"] = pd.to_datetime(c.ymd); c = c[~c.anchor_cd.isin(STADIUM)]
    VLIST[v] = {cat: pick(c, cat, 100) for cat in CATS}
    hcv = pd.concat([city_hours(v, "surge"), city_hours(v, "drop")])
    CITYMAX[v] = hcv.groupby(["ymd", "direction"]).n.max()
def kept(r, cat):
    n = 0
    for v in VLIST:
        if r.scale == "도시 전역":
            n += int(CITYMAX[v].get((r.ymd, r.direction), 0) >= CITY_MIN)
        else:
            x = VLIST[v][cat]; x = x[x.ymd == r.ymd]
            n += int(any(dist[r.anchor_cd].get(a, 99) <= 3 for a in x.anchor_cd))
    return n
SENS = {7: "여러 기준에서도 발견됐어요", 6: "분석 기준에 따라 결과가 달라질 수 있어요", 5: "분석 기준에 따라 결과가 달라질 수 있어요"}

# ---------------- 4. 외부 근거 (공개 출처 사실만) ----------------
facts = pd.read_csv(C_DIR / "facts.csv").set_index("fact_id")
emap = pd.read_csv(ROOT / "src/public/evidence_map_public.csv")
emap["ymd"] = pd.to_datetime(emap.ymd)
def evidence(r):
    """반환: (상태, 내부 수준, 사실 목록, 시각 정보, 출처, 메모). 상태: 판정 완료 / 외부 기록 미검토"""
    m = emap[(emap.ymd == r.ymd) & (emap.anchor == r.anchor)]
    if m.empty:   # 같은 날 3정거장 이내 조사 키(앵커가 이웃 역으로 바뀐 경우)
        near = emap[(emap.ymd == r.ymd)]
        near = near[[dist[r.anchor_cd].get(inv[a][0], 99) <= 3 for a in near.anchor]] if len(near) else near
        m = near
    if m.empty:
        return ("외부 기록 미검토", "", "", "", "", "")
    items = list(m.itertuples())
    real = [i for i in items if i.fact_id in facts.index]
    verified = [i for i in real if facts.loc[i.fact_id, "source_type"] in ("공공기관", "공식예매처", "언론")
                and facts.loc[i.fact_id, "type"] not in ("정기", "달력", "맥락")]
    for i in verified:   # 장소: 사실의 인접역이 사건 앵커에서 2정거장 이내(전역 사실 제외)
        st = str(facts.loc[i.fact_id, "stations"])
        if st not in ("*", "nan", ""):
            assert min(dist[r.anchor_cd].get(c, 99) for c in codes(st.split(";"))) <= 2, (r.ymd, r.anchor, i.fact_id)
    confounded = len(verified) >= 2
    if verified: lvl = "E1" if any(i.time_ok for i in verified) and not confounded else "E2"
    elif [i for i in real]: lvl = "E3"
    else: lvl = "E4"
    # 날짜가 없는 반복 기록(예: 매주 정기 공연)은 기간을 비워 'nan~nan'이 생기지 않게 한다(r5 수정)
    fdesc = lambda f: f"{f['fact']}({f['start']}~{f['end']}, {f['place']})" if pd.notna(f["start"]) and pd.notna(f["end"]) else f"{f['fact']}({f['place']})"
    ftxt = " / ".join(fdesc(facts.loc[i.fact_id]) for i in real)
    ttxt = " / ".join(str(facts.loc[i.fact_id, "time_info"]) if pd.notna(facts.loc[i.fact_id, "time_info"]) else "출처에 시각 없음" for i in real)
    stxt = " ; ".join(str(facts.loc[i.fact_id, "source"]) for i in real if pd.notna(facts.loc[i.fact_id, "source"]))
    note = " / ".join(i.note for i in items if isinstance(i.note, str) and i.note and i.fact_id != "REVIEWED")
    return ("판정 완료", lvl, ftxt, ttxt, stxt, note)
LABEL = {"E1": "확인된 맥락 · 시각까지 일치", "E2": "확인된 맥락 · 날짜·장소만 일치", "E3": "가능한 맥락", "E4": "아직 설명되지 않음", "": "외부 기록 미검토"}

def finalize(df, cat):
    out = []
    for i, r in enumerate(df.itertuples(), 1):
        k = kept(r, cat)
        row = dict(rank=i, event_id=r.event_id, date=r.ymd.date(), dow=r.dow, scale=r.scale, anchor_station=r.anchor,
                   n_local_clusters=r.n_local_clusters, n_station_events=r.n_stations, core_share=r.core_share, stations=r.stations,
                   hour_start=r.h_start, hour_end_excl=r.h_end_excl, obs=r.obs, baseline=r.baseline, excess=r.excess,
                   excess_alight=r.excess_alight, excess_board=r.excess_board, pct_change=r.pct_change, max_abs_z=r.max_abs_z,
                   special_name=r.special_name,
                   sensitivity_kept=k, sensitivity_total=7, sensitivity_label=SENS.get(k, "일부 기준에서만 발견됐어요"),
                   min_n_base=r.min_n_base, low_sample=r.low_sample,
                   sample_label="비교할 같은 조건의 날이 적어요" if r.low_sample else "")
        if cat == "main":
            st, lvl, f, t, s, n = evidence(r)
            row.update(evidence_status=st, evidence_level=lvl, evidence_label=LABEL[lvl], external_facts=f, source_time_info=t, sources=s, notes=n)
        elif cat == "calendar":
            row.update(evidence_status="달력 맥락(E 판정 대상 아님)", evidence_level="", evidence_label=f"특별한 날짜: {r.special_name}",
                       external_facts="", source_time_info="", sources="special_days(공휴일·연휴 목록)", notes="")
        else:
            row.update(evidence_status="외부 기록 미검토", evidence_level="", evidence_label="외부 기록 미검토",
                       external_facts="", source_time_info="", sources="", notes="")
        row["contains_stadium_area_station"] = bool(set(ev[ev.cid.isin([int(x) for x in r.member_cids.split(",")])].sta_cd) & STADIUM)
        out.append(row)
    return pd.DataFrame(out)

FILES = {"main": "a_main_changes_top50.csv", "calendar": "a_special_dates_top20.csv", "drops": "a_decreases_top20.csv"}
PUBL = {}
for cat, fn in FILES.items():
    T = finalize(LISTS[cat], cat); PUBL[cat] = T
    T.to_csv(A_DIR / fn, index=False, encoding="utf-8-sig")
    log(f"[A {cat}] {len(T)}건 | 근거 {T.evidence_label.value_counts().to_dict()} | 일관성 {T.sensitivity_kept.value_counts().sort_index().to_dict()} | 표본 부족 {int(T.low_sample.sum())}")

# 구성 역 사건(공개 목록 사건만) + 시간대별 관측·평소
pub_ids = set(pd.concat([PUBL[c].event_id for c in PUBL]))
comp = cl[cl.event_id.isin(pub_ids)]
st_ev = ev[ev.cid.isin(comp.cid)].merge(comp[["cid", "event_id"]], on="cid")
st_out = st_ev[["event_id", "cid", "sta_cd", "sta_nm", "ymd", "dow", "direction", "h_start", "h_end", "obs", "base", "excess",
                "excess_board", "excess_alight", "max_abs_z", "min_n_base", "low_sample"]].rename(columns={"cid": "local_cluster_id", "base": "baseline"})
st_out["ymd"] = st_out.ymd.dt.date
st_out.to_csv(A_DIR / "a_event_station_components.csv", index=False, encoding="utf-8-sig")
HOUR_LABEL = {h: (f"{h:02d}~{h+1:02d}시" if h < 24 else "24~01시(자정 이후 00~01시)") for h in range(1, 25)}
for h in (1, 2, 3, 4): HOUR_LABEL[h] = f"{h:02d}~{h+1:02d}시(자정 이후, 같은 운영일)"
OP_ORDER = {h: (h - 5 if h >= 5 else h + 19) for h in range(1, 25)}   # 05시 → … → 24시 → 01시 → 04시
OP_ORDER[24] = 19
for h in (1, 2, 3, 4): OP_ORDER[h] = 19 + h
pairs = st_out[["event_id", "sta_cd", "ymd"]].drop_duplicates()
db.register("pairs", pairs.assign(ymd=pd.to_datetime(pairs.ymd)))
H = db.execute(f"""SELECT p.event_id, s.sta_cd, s.sta_nm, s.ymd, s.gubn, s.hour, s.cnt AS obs, s.base_med AS baseline, s.n_base,
                    round(s.z, 2) z, s.is_special FROM pairs p JOIN {tbl('MAIN')} s ON s.sta_cd = p.sta_cd AND s.ymd = p.ymd ORDER BY 1,2,5,6""").df()
H["hour_label"] = H.hour.map(HOUR_LABEL); H["operating_day_order"] = H.hour.map(OP_ORDER); H["ymd"] = pd.to_datetime(H.ymd).dt.date
H.to_csv(A_DIR / "a_event_hourly_obs_vs_baseline.csv", index=False, encoding="utf-8-sig")
city_out = city.assign(ymd=city.ymd.dt.date); city_out.to_csv(A_DIR / "a_citywide_days.csv", index=False, encoding="utf-8-sig")
# 공개 제외(내부 보존): 사직권 지역 사건
SEPARATED.assign(ymd=SEPARATED.ymd.dt.date).drop(columns=["member_cids"]).to_csv(INTERNAL / "a_stadium_area_local_events_withheld.csv", index=False, encoding="utf-8-sig")
log(f"[A] 공개 제외 사직권 지역 사건 {len(SEPARATED)}건(내부 보존). 도시 전역(급증) 날짜:",
    sorted(city[city.direction == "surge"].ymd.dt.strftime("%Y-%m-%d")))

# ---------------- 5. B (행사 전후) ----------------
EVENT_TYPES = ("행사", "공연", "교통", "제도")
fc = facts.reset_index()
known = set()
for r in fc[fc.type.isin(EVENT_TYPES)].itertuples():
    known |= {d.date() for d in pd.date_range(r.start, r.end)}
db.execute("CREATE OR REPLACE TABLE known_event_days AS SELECT unnest(?::DATE[]) AS ymd", [sorted(known)])
SPECIAL = dict(db.execute("SELECT ymd, name FROM special_days").fetchall())

def compare(cds, days, hours, gubn=None):
    hl = ",".join(map(str, hours)); gf = f"AND gubn = '{gubn}'" if gubn else ""
    db.execute(f"""CREATE OR REPLACE TEMP TABLE g AS SELECT ymd, gubn, sum(cnt) v FROM hourly
                   WHERE sta_cd IN ({",".join(map(str, cds))}) AND hour IN ({hl}) {gf} GROUP BY ALL""")
    norm = "SELECT * FROM g WHERE ymd NOT IN (SELECT ymd FROM special_days) AND ymd NOT IN (SELECT ymd FROM known_event_days)"
    df = db.execute(f"""WITH n AS ({norm})
      SELECT t.ymd, t.gubn, t.v obs,
        (SELECT median(v) FROM n WHERE n.gubn=t.gubn AND dayofweek(n.ymd)=dayofweek(t.ymd) AND abs(n.ymd-t.ymd) BETWEEN 1 AND 56) base,
        (SELECT count(*)  FROM n WHERE n.gubn=t.gubn AND dayofweek(n.ymd)=dayofweek(t.ymd) AND abs(n.ymd-t.ymd) BETWEEN 1 AND 56) n_base
      FROM g t WHERE t.ymd IN (SELECT unnest(?::DATE[])) ORDER BY 1,2""", [days]).df()
    nd = db.execute(f"""WITH n AS ({norm})
      SELECT a.gubn, dayofweek(a.ymd) dw, 100*(a.v/median(b.v)-1) pct FROM n a JOIN n b ON a.gubn=b.gubn AND dayofweek(a.ymd)=dayofweek(b.ymd)
      AND abs(a.ymd-b.ymd) BETWEEN 1 AND 56 GROUP BY a.ymd, a.gubn, a.v""").df()
    df["excess"] = df.obs - df.base; df["pct"] = 100 * (df.obs / df.base - 1)
    P = []
    for r in df.itertuples():
        wk = pd.Timestamp(r.ymd).dayofweek >= 5
        dd = nd[(nd.gubn == r.gubn) & (nd.dw.isin([0, 6]) == wk)].pct
        P.append((np.percentile(dd, 95), np.percentile(dd, 5), (dd < r.pct).mean() * 100, len(dd)))
    df["normal_p95_pct"] = [p[0] for p in P]; df["normal_p5_pct"] = [p[1] for p in P]
    df["pct_rank_among_normal"] = [p[2] for p in P]; df["n_normal_days"] = [p[3] for p in P]
    # PRD v0.2 [잠정 USR-5]: 표본 0 → 1층 불가 / 표본 1~7 → 관측은 표시, 2층 '판정 보류' / 8 이상 → 2층 판정
    df["layer1"] = np.where(df.n_base == 0, "비교 자료 없음(판정 불가)", "분석 가능")
    df["layer2"] = np.where(df.n_base == 0, "", np.where(df.n_base < 8, "판정 보류",
                   np.where(df.pct > df.normal_p95_pct, "평소보다 크게 늘었어요",
                   np.where(df.pct < df.normal_p5_pct, "평소보다 크게 줄었어요", "평소 범위 안이었어요"))))
    df["ymd"] = pd.to_datetime(df.ymd)
    return df

ALL_H = list(range(1, 25))
B_FACTS = fc[fc.type.isin(("행사", "공연")) & ~fc.stations.isin(["*"]) & fc.stations.notna()]
DATA_END = pd.Timestamp("2026-09-30")
daily = []
for f in B_FACTS.itertuples():
    if pd.Timestamp(f.start) > DATA_END: continue
    s, e = pd.Timestamp(f.start), pd.Timestamp(f.end)
    days = [d.date() for d in pd.date_range(s - pd.Timedelta(days=7), e + pd.Timedelta(days=7))]
    d = compare(codes(f.stations.split(";")), days, ALL_H)
    d["phase"] = np.where(d.ymd < s, "전", np.where(d.ymd > e, "후", "중"))
    d["rel_day"] = (d.ymd - s).dt.days
    d["event_day_no"] = np.where(d.phase == "중", d.rel_day + 1, np.nan)
    d.insert(0, "fact_id", f.fact_id); d.insert(1, "event", f.fact); d.insert(2, "stations", f.stations)
    d["special_day"] = d.ymd.dt.date.map(lambda x: SPECIAL.get(x, ""))
    d["contains_combined_station"] = any(x in f.stations.split(";") for x in ("수영", "미남"))
    daily.append(d)
BD = pd.concat(daily, ignore_index=True)
small = BD[BD.gubn == "하차"].groupby("fact_id").base.median() < 1000
BD["small_station_group"] = BD.fact_id.map(small)
BD["obs"] = BD.obs.astype(int)
for c in ["base", "excess"]: BD[c] = BD[c].round(0)
for c in ["pct", "normal_p95_pct", "normal_p5_pct", "pct_rank_among_normal"]: BD[c] = BD[c].round(1)
BD["ymd"] = BD.ymd.dt.date
BD.to_csv(B_DIR / "b_event_daily.csv", index=False, encoding="utf-8-sig")
# 행사 기간 요약(3층 '행사 전후도 평소와 달랐음' 판단용 전·후 7일 포함)
def summ(x):
    return pd.Series(dict(n_days=x.ymd.nunique(), obs=int(x.obs.sum()), base=round(x.base.sum()), excess=round(x.excess.sum()),
                          pct=round(100 * (x.obs.sum() / x.base.sum() - 1), 1),
                          days_above_p95=int((x.layer2 == "평소보다 크게 늘었어요").sum()),
                          days_below_p5=int((x.layer2 == "평소보다 크게 줄었어요").sum()),
                          days_on_hold=int((x.layer2 == "판정 보류").sum())))
BP = BD.groupby(["fact_id", "event", "gubn", "phase"]).apply(summ, include_groups=False).reset_index()
BP.to_csv(B_DIR / "b_event_phase_summary.csv", index=False, encoding="utf-8-sig")
log(f"[B] 행사 {BD.fact_id.nunique()}건, 일별 {len(BD)}행")

# 바다축제 2026 정정 확인: 5일차까지 평소 범위 밖, 6일차(8/12)부터 평소 범위
sea = BD[(BD.fact_id == "F03") & (BD.gubn == "하차") & (BD.phase == "중")].set_index("event_day_no")
log("[B] 바다축제 2026 하차 일차별 판정:", {int(k): (v.pct, v.normal_p95_pct, v.layer2) for k, v in sea.iterrows()})
assert sea.loc[5, "layer2"] == "평소보다 크게 늘었어요" and sea.loc[6, "layer2"] == "평소 범위 안이었어요" and sea.loc[7, "layer2"] == "평소 범위 안이었어요"

# 시간 구간 분석: 출처에 시각이 있는 행사만 (규칙은 README)
def slots(t0, t1):
    """[t0, t1) 시각(소수 시간, 24 넘으면 자정 이후)과 겹치는 시간대 → hour 코드(24=00~01시, 1=01~02시)"""
    out = []
    h = int(np.floor(t0))
    while h < t1:
        out.append(24 if h == 24 else (h - 24 if h > 24 else h)); h += 1
    return out
WINDOWS = [  # fact_id, 날짜, 구간 이름, 역, 구분, 시간대, 근거(출처 시각) / 규칙
 ("F01", "2025-11-15", "도착(시작 전 3시간)", "광안;금련산;수영;남천;민락", "하차", slots(11, 14), "부대행사 14시 시작 → [11,14)", "W1"),
 ("F01", "2025-11-15", "귀가(종료 후 3시간)", "광안;금련산;수영;남천;민락", "승차", slots(18, 21), "부대행사 18시 종료 → [18,21)", "W2"),
 ("F03", "2026-08-07", "도착(시작 전 3시간)", "다대포해수욕장;다대포항;낫개", "하차", slots(17, 20), "불꽃쇼 20:00 → [17,20)", "W1"),
 ("F03", "2026-08-07", "귀가(종료 후 3시간)", "다대포해수욕장;다대포항;낫개", "승차", slots(20 + 17 / 60, 23 + 17 / 60), "불꽃쇼 20:00, 17분 → [20:17,23:17)", "W2"),
 ("F04", "2025-08-01", "도착(시작 전 3시간)", "다대포해수욕장;다대포항;낫개", "하차", slots(16.75, 19.75), "불꽃쇼 19:45 → [16:45,19:45)", "W1"),
 ("F11", "2026-07-25", "도착(시작 전 3시간)", "광안;금련산;수영;남천;민락", "하차", slots(17, 20), "20:00 회차 → [17,20)", "W1"),
 ("F11", "2026-07-25", "귀가(종료 후 3시간)", "광안;금련산;수영;남천;민락", "승차", slots(22.2, 25.2), "22:00 회차 12분 이내 → [22:12,01:12)", "W2"),
 ("F13", "2026-09-19", "도착(시작 전 3시간)", "종합운동장;사직", "하차", slots(15, 18), "18:00 시작 → [15,18)", "W1"),
 ("F22", "2025-04-06", "출발권 도착", "센텀시티;벡스코", "하차", [7, 8], "통제 07:00 시작·08시 집결 → 07~09시", "W3"),
 ("F22", "2025-04-06", "도착권 귀가", "광안;금련산", "승차", [10, 11, 12, 13], "출발 뒤 ~ 통제 종료 후 → 10~14시(두 회차 공통 창)", "W3"),
 ("F23", "2026-04-05", "출발권 도착", "센텀시티;벡스코", "하차", [7, 8], "통제 07:00 시작 → 07~09시", "W3"),
 ("F23", "2026-04-05", "도착권 귀가", "광안;금련산", "승차", [10, 11, 12, 13], "출발 뒤 ~ 통제 종료(12:30) 후 → 10~14시(두 회차 공통 창)", "W3"),
]
win = []
for fid, day, part, stas, gubn, hrs, basis, rule in WINDOWS:
    w_ = compare(codes(stas.split(";")), [pd.Timestamp(day).date()], hrs, gubn).iloc[0]
    d_ = compare(codes(stas.split(";")), [pd.Timestamp(day).date()], ALL_H, gubn).iloc[0]
    win.append(dict(fact_id=fid, ymd=day, part=part, stations=stas, gubn=gubn, hours=",".join(map(str, hrs)),
                    hours_label=" ".join(HOUR_LABEL[h] for h in hrs), window_rule=rule, window_basis=basis,
                    win_obs=int(w_.obs), win_base=round(w_.base), win_excess=round(w_.excess), win_pct=round(w_.pct, 1),
                    win_p95=round(w_.normal_p95_pct, 1), win_n_base=int(w_.n_base), win_layer2=w_.layer2,
                    day_obs=int(d_.obs), day_base=round(d_.base), day_excess=round(d_.excess), day_pct=round(d_.pct, 1),
                    day_p95=round(d_.normal_p95_pct, 1), day_layer2=d_.layer2, judgement_differs=w_.layer2 != d_.layer2))
W = pd.DataFrame(win)
# 행사 상세(F05) 기본값 = 카탈로그 관련 역 전체·하루 합계·같은 구분
cat_day = BD[["fact_id", "ymd", "gubn", "pct", "normal_p95_pct", "layer2"]].rename(columns={"pct": "event_day_pct", "normal_p95_pct": "event_day_p95", "layer2": "event_day_layer2"})
cat_day["ymd"] = cat_day.ymd.astype(str)
W = W.merge(cat_day, on=["fact_id", "ymd", "gubn"], how="left")
W["differs_from_event_day"] = W.win_layer2 != W.event_day_layer2
W.to_csv(B_DIR / "b_window_vs_day.csv", index=False, encoding="utf-8-sig")
log("[B] 시간 구간 vs 행사 관련 역 하루 판정 다름:", W[W.differs_from_event_day][["fact_id", "part", "win_pct", "win_p95", "event_day_pct", "event_day_p95"]].to_dict("records"))
log("[B] 하루 합계로 '평소 범위 안'인 행사일(하차):", BD[(BD.phase == "중") & (BD.gubn == "하차") & (BD.layer2 != "평소보다 크게 늘었어요")][["fact_id", "ymd", "pct", "normal_p95_pct"]].to_dict("records"))

# 연도 비교 5개 시리즈 (하차 기준 하루 분석 / 기브앤레이스는 시간 구간)
SERIES = [("부산바다축제", "F04", "F03"), ("강서 낙동강30리 벚꽃축제", "F05", "F06"), ("대저토마토축제", "F07", "F08"),
          ("해운대 모래축제", "F17", "F16"), ("기브앤레이스", "F22", "F23")]
Y = []
for nm, a, b in SERIES:
    for fid in (a, b):
        f = facts.loc[fid]
        if fid in ("F22", "F23"):
            for r in W[W.fact_id == fid].itertuples():
                Y.append(dict(series=nm, fact_id=fid, edition_year=f.start[:4], basis=f"시간 구간: {r.part} {r.gubn} ({r.hours_label})",
                              n_days=1, first_day=r.ymd, first_dow=pd.Timestamp(r.ymd).day_name(), first_obs=r.win_obs, first_base=r.win_base,
                              first_excess=r.win_excess, first_pct=r.win_pct, first_p95=r.win_p95, first_layer2=r.win_layer2,
                              during_excess=r.win_excess, during_pct=r.win_pct, days_above_p95=int(r.win_layer2 == "평소보다 크게 늘었어요"),
                              pre7_pct=None, post7_pct=None))
            continue
        x = BD[(BD.fact_id == fid) & (BD.gubn == "하차")]; d = x[x.phase == "중"].sort_values("ymd"); f0 = d.iloc[0]
        pp = lambda ph: round(100 * (x[x.phase == ph].obs.sum() / x[x.phase == ph].base.sum() - 1), 1)
        Y.append(dict(series=nm, fact_id=fid, edition_year=f.start[:4], basis="하루 합계 하차(관련 역)", n_days=len(d),
                      first_day=f0.ymd, first_dow=pd.Timestamp(f0.ymd).day_name(), first_obs=int(f0.obs), first_base=int(f0.base),
                      first_excess=int(f0.excess), first_pct=f0.pct, first_p95=f0.normal_p95_pct, first_layer2=f0.layer2,
                      during_excess=int(d.excess.sum()), during_pct=round(100 * (d.obs.sum() / d.base.sum() - 1), 1),
                      days_above_p95=int((d.layer2 == "평소보다 크게 늘었어요").sum()), pre7_pct=pp("전"), post7_pct=pp("후")))
YY = pd.DataFrame(Y); YY.to_csv(B_DIR / "b_yoy_comparison.csv", index=False, encoding="utf-8-sig")
log("[B] 연도 비교\n" + YY[["series", "edition_year", "basis", "first_excess", "first_pct", "first_layer2", "during_excess", "days_above_p95", "n_days"]].to_string(index=False))

# 카탈로그 사본(공개 출처 사실) — 이미 public/data/catalog/facts.csv
pd.Series({"known_event_days_excluded_from_B_baseline": sorted(str(d) for d in known)}).to_json(B_DIR / "b_known_event_days.json", force_ascii=False, indent=1)
(INTERNAL / "build_log.txt").write_text("\n".join(LOG), encoding="utf-8")
(PUB / "BUILD_LOG.txt").write_text("\n".join(LOG), encoding="utf-8")
