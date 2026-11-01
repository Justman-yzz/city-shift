"""check_public.py — 공개 폴더 최종 검사 (허용 목록 + 제한 자료·민감정보 검사 + 수치 정합성)

실행(프로젝트 루트): python3 src/public/check_public.py
검사
 1. public/ 안의 파일 = ALLOWLIST.txt 목록 (빠짐·추가 모두 실패)
 2. 금지 열 이름(외부 스포츠 일정 파생 열, 기상 열 등)
 3. 금지 문자열(제한 출처 이름·관련 용어·내부 테이블명). 이 검사 파일 자체는 목록을 담고 있어 문자열 검사에서 제외
 4. 인증키·비밀정보·개인 경로 패턴
 5. 공개 목록에 사직권 7개 역 앵커 지역 사건이 없음
 6. 수치 정합성: 사건 합계 = 구성 역 사건 합계, 구성 역 관측 = 원본 CSV 재독 값(원본이 있을 때)
결과: public/CHECK_RESULT.txt (하나라도 실패하면 종료 코드 1)
"""
import re, sys, pathlib, io
import numpy as np
import pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[2]
PUB = ROOT / "public"
SELF = "code/check_public.py"
R = []
def chk(name, ok, detail=""):
    R.append(f"{'PASS' if ok else 'FAIL'} | {name}" + (f" | {detail}" if detail else "")); return ok

allow = [l.strip() for l in (PUB / "ALLOWLIST.txt").read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
files = sorted(str(p.relative_to(PUB)).replace("\\", "/") for p in PUB.rglob("*") if p.is_file() and ".git/" not in str(p.relative_to(PUB)).replace("\\", "/") + "/")
files = [f for f in files if not f.startswith(".git/")]
extra = sorted(set(files) - set(allow)); missing = sorted(set(allow) - set(files))
chk("허용 목록과 폴더 내용 일치", not extra and not missing, f"추가 {extra} 누락 {missing}")

BAD_COLS = {"crowd", "kbo_auto", "game_day", "game_cond", "recurrent", "played", "first_start", "n_games", "cancel_reason",
            "sajik_game_played", "sajik_start", "sajik_crowd", "sajik_cancel", "sajik_game_start", "precip_mm", "max_temp_c",
            "kma_busan_precip_mm", "net_ratio"}
for f in files:
    if f.endswith(".csv"):
        cols = set(pd.read_csv(PUB / f, nrows=1).columns)
        chk(f"금지 열 없음: {f}", not (cols & BAD_COLS), str(sorted(cols & BAD_COLS)))

TERMS = re.compile(r"KBO|kbo|koreabaseball|관중|롯데|야구|우천|원정|sajik_game|game_cond|game_day|crowd|weather_daily|precip|경기(?!장)")
SECRET = re.compile(r"(?i)service_?key\s*=|DATA_GO_KR_KEY\s*=|authKey\s*=|[A-Za-z0-9%+/]{60,}={0,2}|[\w.+-]+@[\w-]+\.[\w.]+|C:\\\\Users|/home/|/mnt/|/root/")
for f in files:
    if not re.search(r"(\.(csv|md|txt|py|json|html)|^\.gitignore)$", f): chk(f"텍스트 형식 파일만 공개: {f}", False); continue
    t = (PUB / f).read_text(encoding="utf-8-sig")
    if f.endswith(".html"):   # 내장 원천 아카이브(base64)는 문자열 검사에서 빼고 아래에서 값으로 검사
        t = re.sub(r'<script type="text/plain" id="cs-raw">.*?</script>', "", t, flags=re.S)
    if f != SELF:
        m = TERMS.findall(t); chk(f"금지 문자열 없음: {f}", not m, f"검출 {len(m)}건" if m else "")
    if f == SELF: continue   # 검사 패턴 자체를 담은 파일
    s = [x for x in SECRET.findall(t) if x]
    chk(f"키·개인정보·로컬 경로 없음: {f}", not s, f"검출 {len(s)}건" if s else "")

STAD = {"거제", "동래", "만덕", "미남", "사직", "연산", "종합운동장"}
A = PUB / "data" / "a"
comp = pd.read_csv(A / "a_event_station_components.csv")
for f in ["a_main_changes_top50.csv", "a_special_dates_top20.csv", "a_decreases_top20.csv"]:
    t = pd.read_csv(A / f)
    chk(f"사직권 앵커 지역 사건 없음: {f}", not ((t.scale == "지역") & t.anchor_station.isin(STAD)).any())
    g = comp.groupby("event_id")[["obs", "baseline", "excess"]].sum()
    j = t.set_index("event_id")[["obs", "baseline", "excess", "n_local_clusters"]].join(g, rsuffix="_c")
    # 관측은 정확히 일치, 평소·초과는 클러스터별 정수 절삭(중앙값 .5) 때문에 클러스터 수 이내 차이 허용
    chk(f"사건 합계 = 구성 역 합계: {f}", (j.obs == j.obs_c).all() and ((j.excess - j.excess_c).abs() <= j.n_local_clusters).all(),
        f"초과 인원 최대 차 {(j.excess - j.excess_c).abs().max():.1f}명")

# 원본 CSV 재독 대조 (원본은 공개 폴더 밖, 있을 때만)
RAW = {2025: (ROOT / "data/raw/busan_metro_hourly_2025_20251231_utf8.csv", "utf-8-sig"),
       2026: (ROOT / "data/raw/busan_metro_hourly_2026_20260930_cp949.csv", "cp949")}
if all(p.exists() for p, _ in RAW.values()):
    raw = pd.concat([pd.read_csv(io.StringIO(p.read_bytes().decode(e)), dtype={"년월일": str}) for p, e in RAW.values()])
    raw.columns = ["sta_cd", "sta_nm", "ymd", "dow", "gubn", "total"] + [f"h{i:02d}" for i in range(1, 25)]
    raw["ymd"] = pd.to_datetime(raw.ymd).dt.strftime("%Y-%m-%d")
    long = raw.melt(id_vars=["sta_cd", "ymd", "gubn"], value_vars=[f"h{i:02d}" for i in range(1, 25)], var_name="h", value_name="cnt")
    long["hour"] = long.h.str[1:].astype(int)
    hr = pd.read_csv(A / "a_event_hourly_obs_vs_baseline.csv")
    m = hr.merge(long, on=["sta_cd", "ymd", "gubn", "hour"], how="left")
    chk("시간대별 관측 = 원본 CSV(전 행)", (m.obs == m.cnt).all(), f"{len(m):,}행")
    # 구성 역 사건 = 이상 판정 셀(z·증감·배율 기준)의 합 → 시간대별 파일에서 재구성
    hr["delta"] = hr.obs - hr.baseline; hr["ratio"] = (hr.obs + 10) / (hr.baseline + 10)
    bad = 0
    for r in comp.itertuples():
        x = hr[(hr.event_id == r.event_id) & (hr.sta_cd == r.sta_cd) & (hr.ymd == r.ymd) & hr.hour.between(r.h_start, r.h_end)]
        x = x[(x.z >= 4) & (x.delta >= 150) & (x.ratio >= 1.5)] if r.direction == "surge" else x[(x.z <= -4) & (x.delta <= -150) & (x.ratio <= 1 / 1.5)]
        if int(x.obs.sum()) != int(r.obs): bad += 1
    chk("구성 역 사건 관측 = 원본 기반 시간대 셀 재구성(전 건)", bad == 0, f"{len(comp)}건 중 불일치 {bad}")
    # 역 프로필: 원본 '합계' 열·시간 열과 공개 특수일 목록으로 독립 재계산
    S_ = PUB / "data" / "station"
    sp = set(pd.read_csv(PUB / "data/catalog/special_days.csv").ymd)
    rr = raw[~raw.ymd.isin(sp)].copy()
    rr["day_type"] = ["주말" if d >= 5 else "평일" for d in pd.to_datetime(rr.ymd).dt.dayofweek]
    dm = rr.groupby(["sta_cd", "day_type", "gubn"]).total.mean().unstack("gubn")
    sm = pd.read_csv(S_ / "station_profile_summary.csv").set_index(["sta_cd", "day_type"])
    j = sm.join(dm)
    chk("역 프로필 하루 평균 = 원본 합계 열 재계산(112역×2)", ((j.daily_mean_board - j["승차"]).abs() < 0.11).all() and ((j.daily_mean_alight - j["하차"]).abs() < 0.11).all(),
        f"최대 차 {max((j.daily_mean_board - j['승차']).abs().max(), (j.daily_mean_alight - j['하차']).abs().max()):.3f}")
    nd = rr.groupby(["sta_cd", "day_type"]).ymd.nunique()
    chk("역 프로필 집계 일수 = 원본", (sm.n_days == nd.reindex(sm.index)).all(), f"평일 {int(nd.xs('평일', level=1).max())}일·주말 {int(nd.xs('주말', level=1).max())}일")
    hm = rr.melt(id_vars=["sta_cd", "day_type", "gubn"], value_vars=[f"h{i:02d}" for i in range(1, 25)], var_name="h", value_name="cnt")
    hm["hour"] = hm.h.str[1:].astype(int)
    hm = hm.groupby(["sta_cd", "day_type", "gubn", "hour"]).cnt.mean().rename("raw_mean")
    ph = pd.read_csv(S_ / "station_profile_hourly.csv").set_index(["sta_cd", "day_type", "gubn", "hour"]).join(hm)
    chk("역 프로필 시간대 평균 = 원본 재계산(10,752행)", ((ph.mean_cnt - ph.raw_mean).abs() < 0.051).all() and len(ph) == 10752)
    wr = (j.xs("주말", level=1)[["승차", "하차"]].sum(axis=1) / j.xs("평일", level=1)[["승차", "하차"]].sum(axis=1))
    dmax = (sm.xs("평일", level=1).weekend_to_weekday_ratio - wr).abs().max()
    chk("주말/평일 비율 = 원본 재계산(소수 셋째 자리 반올림 허용)", dmax <= 0.001, f"최대 차 {dmax:.5f}")
    pk = ph.reset_index().sort_values(["raw_mean", "operating_day_order"], ascending=[False, True]).groupby(["sta_cd", "day_type", "gubn"]).hour.first()
    chk("정점 시간 = 원본 재계산", all(sm.loc[(c, d), f"peak_hour_{'board' if g == '승차' else 'alight'}"] == v for (c, d, g), v in pk.items()))
    # B: 원본 + 공개 특수일 + 공개 카탈로그 행사일만으로 평소(중앙값)·표본 수·판정 보류 독립 재계산
    import json
    B_ = PUB / "data" / "b"
    bd = pd.read_csv(B_ / "b_event_daily.csv")
    kd = set(json.load(open(B_ / "b_known_event_days.json", encoding="utf-8"))["known_event_days_excluded_from_B_baseline"])
    tot = raw.groupby(["sta_nm", "ymd", "gubn"]).total.sum()
    names_by_code = raw.drop_duplicates("sta_cd").set_index("sta_cd").sta_nm
    daily = raw.groupby(["sta_cd", "ymd", "gubn"]).total.sum().unstack("gubn")
    bad, n = 0, 0
    for fid, g in bd.groupby("fact_id"):
        stas = g.stations.iloc[0].split(";")
        codes_ = [c for c, nm in names_by_code.items() if nm in stas]
        s = daily.loc[daily.index.get_level_values(0).isin(codes_)].groupby(level=1).sum()
        s.index = pd.to_datetime(s.index)
        normal = s[~s.index.strftime("%Y-%m-%d").isin(sp | kd)]
        for r in g.itertuples():
            d = pd.Timestamp(r.ymd)
            dd = np.abs((normal.index - d).days)
            c = normal[(normal.index.dayofweek == d.dayofweek) & (dd >= 1) & (dd <= 56)][r.gubn]
            n += 1
            ok = int(c.count()) == r.n_base and (r.n_base == 0 or abs(c.median() - r.base) <= 0.5) and int(s.loc[d, r.gubn]) == r.obs
            ok = ok and ((r.n_base >= 8) or (r.layer2 == "판정 보류") or (r.n_base == 0 and pd.isna(r.layer2)))
            bad += not ok
    chk("B 일별 관측·평소·표본 수·판정 보류 = 원본 독립 재계산(전 행)", bad == 0, f"{n}행 중 불일치 {bad}")
else:
    chk("원본 CSV 대조", True, "원본 없음 — 건너뜀")

# 배포본(site/index.html) 내장 데이터 = 공개 CSV
SITE = PUB / "site" / "index.html"
if SITE.exists():
    import json, base64, gzip as _gz
    h = SITE.read_text(encoding="utf-8")
    d = json.loads(re.search(r'<script type="application/json" id="cs-data">(.*?)</script>', h, re.S).group(1))
    okA = True
    for k, fn in (("main", "a_main_changes_top50.csv"), ("calendar", "a_special_dates_top20.csv"), ("drops", "a_decreases_top20.csv")):
        t = pd.read_csv(A / fn)
        okA &= [(x["id"], x["obs"], x["base"], x["exc"], x["pct"], x["sens"], x["lab"]) for x in d["A"]["lists"][k]] == \
               [(r.event_id, r.obs, r.baseline, r.excess, r.pct_change, r.sensitivity_kept, r.evidence_label) for r in t.itertuples()]
    chk("배포본 A 목록 내장값 = 공개 CSV(90건)", okA)
    bdx = pd.read_csv(PUB / "data/b/b_event_daily.csv")
    okB = sum(len(v) for v in d["B"]["daily"].values()) == len(bdx) and all(
        [r[4] for r in d["B"]["daily"][fid]] == list(g.sort_values(["ymd", "gubn"]).obs) for fid, g in bdx.groupby("fact_id"))
    chk("배포본 B 일별 내장값 = 공개 CSV", okB)
    st = json.loads(re.search(r'<script type="application/json" id="station-data">(.*?)</script>', h, re.S).group(1))
    sm2 = pd.read_csv(PUB / "data/station/station_profile_summary.csv")
    okS = all(abs(st["p"][str(r.sta_cd)][r.day_type]["db"] - r.daily_mean_board) < 1e-6 for r in sm2.itertuples())
    chk("배포본 역 프로필 내장값 = 공개 CSV", okS)
    if all(p_.exists() for p_, _ in RAW.values()):
        blob = re.search(r'<script type="text/plain" id="cs-raw">(.*?)</script>', h, re.S).group(1).strip()
        u = np.frombuffer(_gz.decompress(base64.b64decode(blob)), dtype=np.uint8); n = len(u) // 2
        z = (u[:n].astype(np.int64) << 8) | u[n:]; v = (z >> 1) ^ -(z & 1)
        v = v.reshape(-1, 638)
        for i in range(7, 638): v[:, i] += v[:, i - 7]
        arr = v.reshape(112, 2, 24, 638)
        codes_ = sorted(raw.sta_cd.unique()); dates_ = sorted(raw.ymd.unique())
        rw = raw.set_index(["sta_cd", "ymd", "gubn"])[[f"h{i:02d}" for i in range(1, 25)]]
        exp = np.zeros_like(arr)
        for (c, dd, g), row in rw.iterrows():
            exp[codes_.index(c), 0 if g == "승차" else 1, :, dates_.index(dd)] = row.values
        chk("배포본 내장 원천 아카이브 = 원본 CSV(3,429,888셀 전부)", bool((arr == exp).all()))
ok = all(x.startswith("PASS") for x in R)
(PUB / "CHECK_RESULT.txt").write_text(f"공개 폴더 검사 결과: {'전부 통과' if ok else '실패 있음'} ({sum(x.startswith('PASS') for x in R)}/{len(R)})\n" + "\n".join(R), encoding="utf-8")
print("\n".join(x for x in R if x.startswith("FAIL")) or "ALL PASS", f"({len(R)} checks)")
sys.exit(0 if ok else 1)
