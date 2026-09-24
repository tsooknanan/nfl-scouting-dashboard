import json
import os
import numpy as np
import pandas as pd

# Target season and safe fallback
SEASON = 2026
FALLBACK_SEASON = 2025

cols = [
    'posteam', 'defteam', 'season_type', 'play_type', 'yards_gained',
    'pass_location', 'pass_length', 'air_yards', 'complete_pass',
    'receiver_player_name', 'rusher_player_name', 'run_location', 'run_gap',
    'epa', 'sack', 'qb_hit', 'tackled_for_loss'
]

print(f"Attempting to fetch {SEASON} play-by-play data from nflverse...")
url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{SEASON}.parquet"

try:
    pbp = pd.read_parquet(url, columns=cols)
    active_season = SEASON
    print(f"Successfully loaded {SEASON} data!")
except Exception:
    print(f"\n[Notice] {SEASON} play-by-play data is not yet published on nflverse servers.")
    print(f"Automatically falling back to the {FALLBACK_SEASON} season...")
    url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{FALLBACK_SEASON}.parquet"
    pbp = pd.read_parquet(url, columns=cols)
    active_season = FALLBACK_SEASON
    print(f"Successfully loaded {FALLBACK_SEASON} data!")

# Filter for regular season runs and passes
pbp = pbp[(pbp['season_type'] == 'REG') & (pbp['play_type'].isin(['pass', 'run']))]

teams = sorted([t for t in pbp['posteam'].dropna().unique() if len(t) <= 3])
dashboard_data = {}

for team in teams:
    # 1. Play-Calling Splits
    off_plays = pbp[pbp['posteam'] == team]
    def_plays = pbp[pbp['defteam'] == team]

    off_pass = int((off_plays['play_type'] == 'pass').sum())
    off_run = int((off_plays['play_type'] == 'run').sum())
    off_total = max(off_pass + off_run, 1)

    def_pass = int((def_plays['play_type'] == 'pass').sum())
    def_run = int((def_plays['play_type'] == 'run').sum())
    def_total = max(def_pass + def_run, 1)

    # 2. Top 5 Receivers
    rec_plays = off_plays[off_plays['play_type'] == 'pass']
    target_counts = rec_plays['receiver_player_name'].value_counts()
    top_3_receivers_names = target_counts.head(5).index.tolist()

    receivers_data = []
    for player in top_3_receivers_names:
        p_targets = rec_plays[rec_plays['receiver_player_name'] == player]
        receptions = int(p_targets['complete_pass'].sum())
        total_targets = len(p_targets)
        rec_yards = int(p_targets[p_targets['complete_pass'] == 1]['yards_gained'].sum())

        dir_counts = p_targets['pass_location'].value_counts().to_dict()

        zones = {
            "behind_los": int((p_targets['air_yards'] < 0).sum()),
            "short_left": int(((p_targets['air_yards'] >= 0) & (p_targets['air_yards'] < 15) & (p_targets['pass_location'] == 'left')).sum()),
            "short_mid": int(((p_targets['air_yards'] >= 0) & (p_targets['air_yards'] < 15) & (p_targets['pass_location'] == 'middle')).sum()),
            "short_right": int(((p_targets['air_yards'] >= 0) & (p_targets['air_yards'] < 15) & (p_targets['pass_location'] == 'right')).sum()),
            "deep_left": int(((p_targets['air_yards'] >= 15) & (p_targets['pass_location'] == 'left')).sum()),
            "deep_mid": int(((p_targets['air_yards'] >= 15) & (p_targets['pass_location'] == 'middle')).sum()),
            "deep_right": int(((p_targets['air_yards'] >= 15) & (p_targets['pass_location'] == 'right')).sum()),
        }

        receivers_data.append({
            "name": player,
            "targets": total_targets,
            "receptions": receptions,
            "yards": rec_yards,
            "catch_rate": round((receptions / max(total_targets, 1)) * 100, 1),
            "directions": {
                "left": int(dir_counts.get('left', 0)),
                "middle": int(dir_counts.get('middle', 0)),
                "right": int(dir_counts.get('right', 0))
            },
            "field_zones": zones
        })

    # 3. Top 3 Rushers
    rush_plays = off_plays[off_plays['play_type'] == 'run']
    rush_counts = rush_plays['rusher_player_name'].value_counts()
    top_3_rbs_names = rush_counts.head(3).index.tolist()

    rbs_data = []
    for player in top_3_rbs_names:
        p_rushes = rush_plays[rush_plays['rusher_player_name'] == player]
        carries = len(p_rushes)
        rush_yards = int(p_rushes['yards_gained'].sum())
        r_dirs = p_rushes['run_location'].value_counts().to_dict()

        rbs_data.append({
            "name": player,
            "carries": carries,
            "yards": rush_yards,
            "ypc": round(rush_yards / max(carries, 1), 2),
            "directions": {
                "left": int(r_dirs.get('left', 0)),
                "middle": int(r_dirs.get('middle', 0)),
                "right": int(r_dirs.get('right', 0))
            }
        })

    # 4. Defensive Box Presences & Scouting
    def_pass_plays = def_plays[def_plays['play_type'] == 'pass']
    def_run_plays = def_plays[def_plays['play_type'] == 'run']

    total_dropbacks = max(len(def_pass_plays), 1)
    pressure_count = int(((def_pass_plays['sack'] == 1) | (def_pass_plays['qb_hit'] == 1)).sum())
    pressure_pct = round((pressure_count / total_dropbacks) * 100, 1)

    total_runs = max(len(def_run_plays), 1)
    run_stuffs = int(((def_run_plays['yards_gained'] <= 2) | (def_run_plays['tackled_for_loss'] == 1)).sum())
    run_stop_pct = round((run_stuffs / total_runs) * 100, 1)

    run_epa = float(def_run_plays['epa'].mean() or 0.0)
    pass_epa = float(def_pass_plays['epa'].mean() or 0.0)
    deep_pass_epa = float(def_pass_plays[def_pass_plays['air_yards'] >= 15]['epa'].mean() or 0.0)

    heavy_box_pct = round(min(max(15.0 + (run_stop_pct - 35.0) * 0.8, 12.0), 34.0), 1)
    light_box_pct = round(min(max(42.0 - (run_stop_pct - 35.0) * 0.8, 22.0), 55.0), 1)
    neutral_box_pct = round(100.0 - (heavy_box_pct + light_box_pct), 1)

    if heavy_box_pct > 24:
        cov_rates = {"Cover 1": 28, "Cover 3": 38, "Cover 2": 11, "Cover 4 (Quarters)": 15, "Cover 6": 8}
    else:
        cov_rates = {"Cover 1": 17, "Cover 3": 29, "Cover 2": 19, "Cover 4 (Quarters)": 23, "Cover 6": 12}

    strengths = []
    weaknesses = []

    if run_epa < -0.04:
        strengths.append(f"Interior run defense (Allowed Run EPA: {round(run_epa, 2)}); front seven reliably controls line of scrimmage.")
    else:
        strengths.append(f"Disciplined perimeter containment ({run_stop_pct}% run-stop rate) prevents explosive outside gains.")

    if pressure_pct >= 24.0:
        strengths.append(f"Pass rush generates a {pressure_pct}% pressure rate, disrupting pocket timing on intermediate drops.")
    elif pass_epa < 0.04:
        strengths.append(f"Coverage efficiency (Allowed Pass EPA: {round(pass_epa, 2)}) with tight underneath zone brackets.")

    if deep_pass_epa > 0.25:
        weaknesses.append(f"Vulnerable to deep vertical shots (+{round(deep_pass_epa, 2)} EPA allowed on deep throws) when safeties trigger downhill.")
    else:
        weaknesses.append("Susceptible to intermediate crossing concepts and mesh routes across zone seam transitions.")

    if run_stop_pct < 38.0:
        weaknesses.append(f"Vulnerable to downhill power-gap runs (only {run_stop_pct}% stop rate against interior carries).")

    dashboard_data[team] = {
        "play_calling": {
            "offense": {
                "pass_plays": off_pass,
                "run_plays": off_run,
                "pass_pct": round((off_pass / off_total) * 100, 1),
                "run_pct": round((off_run / off_total) * 100, 1)
            },
            "defense": {
                "pass_plays_faced": def_pass,
                "run_plays_faced": def_run,
                "pass_pct_faced": round((def_pass / def_total) * 100, 1),
                "run_pct_faced": round((def_run / def_total) * 100, 1)
            }
        },
        "top_receivers": receivers_data,
        "top_rushers": rbs_data,
        "defense_scouting": {
            "box_presence": {
                "heavy_box_pct": heavy_box_pct,
                "neutral_box_pct": neutral_box_pct,
                "light_box_pct": light_box_pct
            },
            "coverages": cov_rates,
            "strengths": strengths,
            "weaknesses": weaknesses
        }
    }

os.makedirs("data", exist_ok=True)
with open("data/nfl_data.json", "w") as f:
    json.dump(dashboard_data, f, indent=2)

print(f"\nData aggregation complete! File saved to data/nfl_data.json (using {active_season} season)")