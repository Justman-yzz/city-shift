"""build_station_profiles.py — 역 프로필 · STATION RIVALS 공개 데이터 (부산교통공사 원본 + 공휴일 목록만 사용)

실행(프로젝트 루트, build_public_r1.py 실행 뒤): python3 src/public/build_station_profiles.py
입력: data/public_build.duckdb 의 hourly_flag, special_days
산출: public/data/station/station_profile_hourly.csv, station_profile_summary.csv, stations.csv
      public/data/catalog/special_days.csv (검산·재현용)

정의 (모두 원본에서 다시 셀 수 있는 값)
 기간      : 2025-01-01 ~ 2026-09-30, 특수일(공휴일·연휴 블록) 제외
 평일/주말 : 월~금 / 토·일
 시간대 평균 : 해당 역·구분·시간대의 일별 인원 평균 (집계 일수 n_days)
 하루 중 비중 : 시간대 평균 ÷ 같은 역·구분·요일 유형의 하루 평균
 하루 평균 : 시간대 평균의 합 (= 일별 합계의 평균)
 정점 시간 : 시간대 평균이 가장 큰 시간대(같으면 운영일 순서상 이른 시간)
 주말/평일 비율 : 주말 하루 평균(승차+하차) ÷ 평일 하루 평균(승차+하차)
시간: hour 1~24 = 원본 열 01시-02시 … 24시-01시. 자정 이후(24, 1~4)는 같은 운영일. 운영일 순서 05 → 24 → 01 → 04.
"""
import pandas as pd
from pub_common import connect, PUB

OUT = PUB / "data" / "station"; OUT.mkdir(parents=True, exist_ok=True)
db = connect()
COMBINED = {"수영": "2·3호선 합산", "미남": "3·4호선 합산"}

st = db.execute("SELECT DISTINCT sta_cd, sta_nm FROM hourly_flag ORDER BY 1").df()
line = lambda c: 1 if c < 200 else c // 100
dup = st.sta_nm.value_counts()
st["line"] = st.sta_cd.map(line)
st["line_label"] = [COMBINED.get(n, f"{l}호선") for n, l in zip(st.sta_nm, st.line)]
st["display_name"] = [f"{n} ({ll})" if dup[n] > 1 else n for n, ll in zip(st.sta_nm, st.line_label)]
st["combined_lines"] = st.sta_nm.isin(COMBINED)
st.to_csv(OUT / "stations.csv", index=False, encoding="utf-8-sig")

h = db.execute("""
SELECT sta_cd, CASE WHEN dow_n BETWEEN 1 AND 5 THEN '평일' ELSE '주말' END AS day_type, gubn, hour,
       avg(cnt) AS mean_cnt, count(DISTINCT ymd) AS n_days
FROM hourly_flag WHERE NOT is_special GROUP BY ALL""").df()
h = h.merge(st[["sta_cd", "sta_nm", "display_name", "line_label"]], on="sta_cd")
h["daily_mean"] = h.groupby(["sta_cd", "day_type", "gubn"]).mean_cnt.transform("sum")
h["share_of_day"] = (h.mean_cnt / h.daily_mean).round(5)
OP = {x: (x - 5 if x >= 5 else x + 19) for x in range(1, 25)}; OP[24] = 19
for x in (1, 2, 3, 4): OP[x] = 19 + x
LBL = {x: f"{x:02d}~{x+1:02d}시" for x in range(5, 24)}; LBL[24] = "24~01시(자정 이후)"
for x in (1, 2, 3, 4): LBL[x] = f"{x:02d}~{x+1:02d}시(자정 이후)"
h["operating_day_order"] = h.hour.map(OP); h["hour_label"] = h.hour.map(LBL)
h["mean_cnt"] = h.mean_cnt.round(1); h["daily_mean"] = h.daily_mean.round(1)
h = h.sort_values(["sta_cd", "day_type", "gubn", "operating_day_order"])
h[["sta_cd", "sta_nm", "display_name", "line_label", "day_type", "gubn", "hour", "hour_label", "operating_day_order",
   "mean_cnt", "share_of_day", "daily_mean", "n_days"]].to_csv(OUT / "station_profile_hourly.csv", index=False, encoding="utf-8-sig")

rows = []
for (cd, dt), g in h.groupby(["sta_cd", "day_type"]):
    r = dict(sta_cd=cd, sta_nm=g.sta_nm.iloc[0], display_name=g.display_name.iloc[0], line_label=g.line_label.iloc[0], day_type=dt,
             n_days=int(g.n_days.max()))
    for gb, k in (("승차", "board"), ("하차", "alight")):
        x = g[g.gubn == gb].sort_values(["mean_cnt", "operating_day_order"], ascending=[False, True])
        p = x.iloc[0]
        r[f"daily_mean_{k}"] = float(x.daily_mean.iloc[0])   # 반올림 전 합계를 한 번만 반올림한 값
        r[f"peak_hour_{k}"] = int(p.hour); r[f"peak_label_{k}"] = p.hour_label
        r[f"peak_mean_{k}"] = float(p.mean_cnt); r[f"peak_share_{k}"] = float(p.share_of_day)
    r["daily_mean_total"] = round(r["daily_mean_board"] + r["daily_mean_alight"], 1)
    rows.append(r)
S = pd.DataFrame(rows)
wk = S[S.day_type == "평일"].set_index("sta_cd").daily_mean_total; we = S[S.day_type == "주말"].set_index("sta_cd").daily_mean_total
S["weekend_to_weekday_ratio"] = S.sta_cd.map((we / wk).round(3))
S.to_csv(OUT / "station_profile_summary.csv", index=False, encoding="utf-8-sig")
db.execute("SELECT ymd, name FROM special_days ORDER BY 1").df().to_csv(PUB / "data" / "catalog" / "special_days.csv", index=False, encoding="utf-8-sig")
print(f"역 {S.sta_cd.nunique()}개, 평일 {int(S[S.day_type=='평일'].n_days.max())}일, 주말 {int(S[S.day_type=='주말'].n_days.max())}일, 시간대 행 {len(h)}")
print(S[S.day_type == "평일"].sort_values("daily_mean_total", ascending=False).head(3)[["display_name", "daily_mean_total", "peak_label_board", "peak_label_alight", "weekend_to_weekday_ratio"]].to_string(index=False))
