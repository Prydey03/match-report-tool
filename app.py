from __future__ import annotations

from io import BytesIO
import json
import textwrap
from typing import Iterable
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import streamlit as st
from mplsoccer import Pitch
from statsbombpy import sb


st.set_page_config(page_title="StatsBomb Coaching Report", page_icon="⚽", layout="wide")

TEAM_COLOUR_MAP = {}
PITCH = Pitch(pitch_type="statsbomb", pitch_color="#2b7a3d", line_color="white")

TEMPLATE_COLUMNS = [
    "match_id", "match_date", "home_team", "away_team", "period", "minute", "second",
    "team", "player", "jersey_number", "position", "event_type", "outcome", "recipient",
    "x", "y", "end_x", "end_y", "xg", "possession_team", "duration", "card", "notes",
    "home_colour", "away_colour",
]

COLUMN_ALIASES = {
    "match_id": ["match_id", "match id", "game_id", "fixture_id"],
    "match_date": ["match_date", "match date", "date", "game_date"],
    "home_team": ["home_team", "home team", "home"],
    "away_team": ["away_team", "away team", "away", "opposition", "opponent"],
    "period": ["period", "half", "match_period"],
    "minute": ["minute", "min", "match_minute", "start_minute", "start"],
    "second": ["second", "sec", "match_second", "start_second"],
    "team": ["team", "team_name", "side", "squad"],
    "player": ["player", "player_name", "athlete", "name"],
    "jersey_number": ["jersey_number", "shirt_number", "squad_number", "number", "no"],
    "position": ["position", "player_position", "role"],
    "event_type": ["event_type", "event", "type", "action", "code", "category"],
    "outcome": ["outcome", "result", "event_outcome", "success"],
    "recipient": ["recipient", "pass_recipient", "receiver", "target_player"],
    "x": ["x", "start_x", "location_x", "pos_x"],
    "y": ["y", "start_y", "location_y", "pos_y"],
    "end_x": ["end_x", "x2", "finish_x", "target_x"],
    "end_y": ["end_y", "y2", "finish_y", "target_y"],
    "xg": ["xg", "expected_goals", "shot_xg"],
    "possession_team": ["possession_team", "team_in_possession"],
    "duration": ["duration", "event_duration", "length"],
    "card": ["card", "card_type", "discipline"],
    "notes": ["notes", "note", "comment", "descriptor", "label"],
    "home_colour": ["home_colour", "home_color", "home team colour", "home team color"],
    "away_colour": ["away_colour", "away_color", "away team colour", "away team color"],
}

EVENT_NAMES = {
    "pass": "Pass", "shot": "Shot", "goal": "Shot", "pressure": "Pressure",
    "press": "Pressure", "ball recovery": "Ball Recovery", "recovery": "Ball Recovery",
    "interception": "Interception", "block": "Block", "duel": "Duel", "tackle": "Duel",
    "foul": "Foul Committed", "foul committed": "Foul Committed",
    "substitution": "Substitution", "sub": "Substitution", "carry": "Carry",
    "dribble": "Dribble", "clearance": "Clearance", "offside": "Offside",
}
CANONICAL_EVENTS = ["Pass", "Shot", "Pressure", "Ball Recovery", "Interception", "Block", "Duel",
                    "Foul Committed", "Substitution", "Carry", "Dribble", "Clearance", "Offside", "Unknown"]


def canonical_event_name(value) -> str:
    text = str(value).strip().lower()
    if text in EVENT_NAMES:
        return EVENT_NAMES[text]
    simplified = text.replace("_", " ").replace("-", " ").replace(".", " ")
    for alias, canonical in sorted(EVENT_NAMES.items(), key=lambda item: len(item[0]), reverse=True):
        if alias in simplified:
            return canonical
    return str(value).strip().title() or "Unknown"


# -----------------------------------------------------------------------------
# Data loading and safe helpers
# -----------------------------------------------------------------------------
@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def load_competitions() -> pd.DataFrame:
    return sb.competitions()


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def load_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    return sb.matches(competition_id=competition_id, season_id=season_id)


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def load_match(match_id: int):
    return sb.events(match_id=match_id), sb.lineups(match_id=match_id)


def custom_template() -> pd.DataFrame:
    rows = [
        ["MATCH-001", "2026-09-15", "Home Team", "Away Team", 1, 3, 12, "Home Team",
         "Alex Example", 8, "Centre Midfield", "Pass", "Complete", "Sam Example",
         42, 38, 61, 44, "", "Home Team", 1.2, "", "Delete example rows before entering your match",
         "#138A36", "#1F3A5F"],
        ["MATCH-001", "2026-09-15", "Home Team", "Away Team", 1, 8, 25, "Away Team",
         "Taylor Example", 9, "Striker", "Shot", "Saved", "", 104, 36, "", "", 0.18,
         "Away Team", 0.8, "", "", "#138A36", "#1F3A5F"],
        ["MATCH-001", "2026-09-15", "Home Team", "Away Team", 1, 19, 4, "Home Team",
         "Jordan Example", 4, "Centre Back", "Interception", "Won", "", 67, 52, "", "", "",
         "Away Team", 0.0, "", "", "#138A36", "#1F3A5F"],
    ]
    return pd.DataFrame(rows, columns=TEMPLATE_COLUMNS)


def template_dictionary() -> pd.DataFrame:
    required = {"minute", "team", "event_type"}
    descriptions = {
        "period": "1 or 2 (extra-time periods may use 3 or 4)",
        "minute": "Elapsed match minute as a number",
        "second": "Seconds within the minute, 0–59",
        "event_type": "Pass, Shot, Pressure, Recovery, Interception, Duel, Foul, Substitution, etc.",
        "outcome": "Complete/Incomplete for passes; Goal/Saved/Off Target/etc. for shots",
        "x": "Event start, scaled 0–120 from own goal to opposition goal",
        "y": "Event start, scaled 0–80 from left touchline to right touchline",
        "end_x": "Event end x-coordinate, primarily for passes and carries",
        "end_y": "Event end y-coordinate",
        "xg": "Expected-goals value for a shot, normally between 0 and 1",
        "duration": "Event duration in seconds",
        "home_colour": "Optional home-team colour as a hex code, for example #138A36",
        "away_colour": "Optional away-team colour as a hex code, for example #1F3A5F",
    }
    return pd.DataFrame({
        "field": TEMPLATE_COLUMNS,
        "required": ["Yes" if field in required else "No" for field in TEMPLATE_COLUMNS],
        "description": [descriptions.get(field, field.replace("_", " ").title()) for field in TEMPLATE_COLUMNS],
    })


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    payload = uploaded_file.getvalue()
    if name.endswith(".csv") or name.endswith(".tsv"):
        separator = "\t" if name.endswith(".tsv") else None
        return pd.read_csv(BytesIO(payload), sep=separator, engine="python")
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(BytesIO(payload))
    if name.endswith(".json"):
        parsed = json.loads(payload.decode("utf-8-sig"))
        if isinstance(parsed, dict):
            for key in ["events", "data", "rows", "instances"]:
                if isinstance(parsed.get(key), list):
                    parsed = parsed[key]
                    break
        return pd.json_normalize(parsed)
    if name.endswith(".xml"):
        root = ET.fromstring(payload)
        rows = []
        for instance in root.findall(".//instance"):
            row = {
                "event_type": instance.findtext("code") or instance.findtext("name"),
                "start": instance.findtext("start"),
                "end": instance.findtext("end"),
            }
            labels = []
            for label in instance.findall(".//label"):
                group, value = label.findtext("group"), label.findtext("text")
                if group and value:
                    row[group] = value
                    labels.append(f"{group}: {value}")
            row["notes"] = "; ".join(labels)
            rows.append(row)
        if not rows:
            raise ValueError("No <instance> event records were found in the XML file.")
        return pd.DataFrame(rows)
    raise ValueError("Supported uploads are CSV, TSV, Excel, JSON and Sportscode-style XML.")


def guess_mapping(columns: Iterable[str]) -> dict[str, str | None]:
    columns = list(columns)
    normalised = {str(column).strip().lower().replace("-", "_"): column for column in columns}
    mapping = {}
    for field, aliases in COLUMN_ALIASES.items():
        match = None
        for alias in aliases:
            key = alias.strip().lower().replace("-", "_")
            if key in normalised:
                match = normalised[key]
                break
        mapping[field] = match
    return mapping


def column_values(raw: pd.DataFrame, mapping: dict, field: str, default=np.nan) -> pd.Series:
    column = mapping.get(field)
    return raw[column] if column in raw.columns else pd.Series(default, index=raw.index)


def valid_hex_colour(value, fallback: str) -> str:
    text = str(value).strip()
    if len(text) == 7 and text.startswith("#"):
        try:
            int(text[1:], 16)
            return text.upper()
        except ValueError:
            pass
    return fallback


def scale_coordinates(values: pd.Series, axis: str, system: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if system == "0–1":
        return numeric * (120 if axis == "x" else 80)
    if system == "0–100":
        return numeric * (1.2 if axis == "x" else .8)
    return numeric


def normalise_uploaded_events(raw: pd.DataFrame, mapping: dict, coordinate_system: str,
                              home_team: str, away_team: str, flip_away: bool,
                              event_value_map: dict | None = None) -> pd.DataFrame:
    events = pd.DataFrame(index=raw.index)
    events["id"] = [f"upload-{index}" for index in range(len(raw))]
    events["index"] = np.arange(len(raw))
    events["minute"] = pd.to_numeric(column_values(raw, mapping, "minute"), errors="coerce")
    if events.minute.isna().all() and mapping.get("minute"):
        events["minute"] = pd.to_numeric(column_values(raw, mapping, "minute"), errors="coerce") / 60
    events["minute"] = events.minute.fillna(0).astype(int)
    events["second"] = pd.to_numeric(column_values(raw, mapping, "second", 0), errors="coerce").fillna(0).astype(int)
    events["period"] = pd.to_numeric(column_values(raw, mapping, "period", 1), errors="coerce").fillna(1).astype(int)
    events["team"] = column_values(raw, mapping, "team", home_team).fillna(home_team).astype(str).str.strip()
    events["player"] = column_values(raw, mapping, "player")
    events["pass_recipient"] = column_values(raw, mapping, "recipient")
    raw_types = column_values(raw, mapping, "event_type", "Unknown").fillna("Unknown").astype(str).str.strip()
    events["type"] = raw_types.map(lambda value: (event_value_map or {}).get(value, canonical_event_name(value)))
    events["substitution_replacement"] = column_values(raw, mapping, "recipient").where(events.type.eq("Substitution"))
    outcome = column_values(raw, mapping, "outcome").astype("object")
    complete_terms = {"complete", "completed", "successful", "success", "won", "true", "1", "nan"}
    events["pass_outcome"] = outcome.where(~outcome.astype(str).str.lower().isin(complete_terms), np.nan)
    events["shot_outcome"] = outcome.astype(str).str.title().replace("Nan", np.nan)
    goal_rows = raw_types.str.lower().eq("goal")
    events.loc[goal_rows, "shot_outcome"] = "Goal"
    events["shot_statsbomb_xg"] = pd.to_numeric(column_values(raw, mapping, "xg", 0), errors="coerce").fillna(0)
    events["possession_team"] = column_values(raw, mapping, "possession_team")
    events["possession_team"] = events.possession_team.fillna(events.team)
    events["possession"] = events.team.ne(events.team.shift()).cumsum()
    events["duration"] = pd.to_numeric(column_values(raw, mapping, "duration", 0), errors="coerce").fillna(0)
    events["bad_behaviour_card"] = column_values(raw, mapping, "card")
    events["foul_committed_card"] = column_values(raw, mapping, "card").where(events.type.eq("Foul Committed"))
    events["notes"] = column_values(raw, mapping, "notes")
    for field, axis in [("x", "x"), ("y", "y"), ("end_x", "x"), ("end_y", "y")]:
        events[field] = scale_coordinates(column_values(raw, mapping, field), axis, coordinate_system)
    if flip_away:
        away = events.team.eq(away_team)
        events.loc[away, "x"] = 120 - events.loc[away, "x"]
        events.loc[away, "y"] = 80 - events.loc[away, "y"]
        events.loc[away, "end_x"] = 120 - events.loc[away, "end_x"]
        events.loc[away, "end_y"] = 80 - events.loc[away, "end_y"]
    events["location"] = events.apply(lambda row: [row.x, row.y] if pd.notna(row.x) and pd.notna(row.y) else np.nan, axis=1)
    events["pass_end_location"] = events.apply(
        lambda row: [row.end_x, row.end_y] if pd.notna(row.end_x) and pd.notna(row.end_y) else np.nan, axis=1)
    return events


def uploaded_lineups(raw: pd.DataFrame, mapping: dict, teams: list[str]) -> dict[str, pd.DataFrame]:
    frames = {}
    for team in teams:
        mask = column_values(raw, mapping, "team", team).astype(str).str.strip().eq(team)
        frame = pd.DataFrame({
            "player_name": column_values(raw, mapping, "player")[mask],
            "jersey_number": pd.to_numeric(column_values(raw, mapping, "jersey_number")[mask], errors="coerce"),
            "position_name": column_values(raw, mapping, "position")[mask],
        }).dropna(subset=["player_name"]).drop_duplicates("player_name")
        frame["positions"] = frame.position_name.apply(
            lambda value: [{"position": value if pd.notna(value) else "—", "start_reason": "Starting XI", "from": "00:00:00"}])
        frames[team] = frame.drop(columns="position_name").reset_index(drop=True)
    return frames


def series_or_default(df: pd.DataFrame, column: str, default=np.nan) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(default, index=df.index)


def safe_xy(value):
    return value[:2] if isinstance(value, (list, tuple)) and len(value) >= 2 else (np.nan, np.nan)


def add_xy(df: pd.DataFrame, source="location", prefix="") -> pd.DataFrame:
    result = df.copy()
    if source not in result.columns:
        result[f"{prefix}x"] = np.nan
        result[f"{prefix}y"] = np.nan
        return result
    coordinates = result[source].apply(safe_xy)
    result[[f"{prefix}x", f"{prefix}y"]] = pd.DataFrame(coordinates.tolist(), index=result.index)
    return result


def format_table(df: pd.DataFrame, hide_index: bool = False):
    display = df.copy()
    if display.index.name:
        display = display.reset_index()
    else:
        display = display.reset_index(drop=True)
    renamed = display.rename(columns=lambda value: str(value).replace("_", " ").title())
    formats = {}
    for column in renamed.columns:
        if pd.api.types.is_float_dtype(renamed[column]):
            name = str(column).lower()
            formats[column] = "{:.2f}" if "xg" in name else "{:.1f}"
    styled = (
        renamed.style.format(formats, na_rep="—")
        .set_properties(**{"text-align": "center"})
        .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])
    )
    return styled.hide(axis="index")


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def figure_png(fig) -> bytes:
    output = BytesIO()
    fig.savefig(output, format="png", dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    return output.getvalue()


def draw_and_show(fig, filename: str | None = None, label: str = "Download chart PNG"):
    st.pyplot(fig, use_container_width=True)
    if filename:
        st.download_button(label, figure_png(fig), filename, "image/png", key=f"png_{filename}")
    plt.close(fig)


def add_direction_label(fig):
    fig.text(.5, .015, "Attacking direction  →", ha="center", fontsize=9, weight="bold")


def match_end_minute(events: pd.DataFrame) -> int:
    if events.empty or "minute" not in events:
        return 90
    return max(90, int(pd.to_numeric(events["minute"], errors="coerce").max()))


def match_state_mask(events: pd.DataFrame, focus_team: str, state: str) -> pd.Series:
    if state == "All states" or events.empty:
        return pd.Series(True, index=events.index)
    ordered = events.sort_values([c for c in ["minute", "second", "index"] if c in events.columns])
    focus_score = opponent_score = 0
    states = {}
    for index, event in ordered.iterrows():
        current = "Drawing" if focus_score == opponent_score else ("Winning" if focus_score > opponent_score else "Losing")
        states[index] = current
        if event.get("type") == "Shot" and event.get("shot_outcome") == "Goal":
            if event.get("team") == focus_team:
                focus_score += 1
            else:
                opponent_score += 1
    return pd.Series(states).reindex(events.index).eq(state)


def count_type(events: pd.DataFrame, event_type: str, team: str) -> int:
    return int(((series_or_default(events, "type") == event_type) &
                (series_or_default(events, "team") == team)).sum())


def possession_estimate(events: pd.DataFrame, teams: list[str]) -> pd.Series:
    """Estimate possession from event duration; fall back to possession sequences.

    StatsBomb open event data has possession identifiers, but not an official
    possession percentage. This is intentionally presented as an estimate.
    """
    valid = events[series_or_default(events, "possession_team").isin(teams)].copy()
    duration = pd.to_numeric(series_or_default(valid, "duration", 0), errors="coerce").fillna(0)
    timed = valid.assign(_duration=duration).groupby("possession_team")["_duration"].sum()
    if timed.sum() > 0:
        return (timed / timed.sum() * 100).reindex(teams).fillna(0)
    sequences = valid.groupby("possession_team")["possession"].nunique()
    return (sequences / sequences.sum() * 100).reindex(teams).fillna(0)


def completed_passes(events: pd.DataFrame, team: str | None = None) -> pd.DataFrame:
    mask = series_or_default(events, "type").eq("Pass") & series_or_default(events, "pass_outcome").isna()
    if team:
        mask &= series_or_default(events, "team").eq(team)
    return events[mask].copy()


def pass_metrics(events: pd.DataFrame, team: str) -> dict:
    passes = events[series_or_default(events, "type").eq("Pass") & series_or_default(events, "team").eq(team)].copy()
    passes = add_xy(add_xy(passes, "pass_end_location", "end_"))
    completed = series_or_default(passes, "pass_outcome").isna()
    start_goal_distance = np.sqrt((120 - passes.x) ** 2 + (40 - passes.y) ** 2)
    end_goal_distance = np.sqrt((120 - passes.end_x) ** 2 + (40 - passes.end_y) ** 2)
    progressive = completed & (end_goal_distance <= start_goal_distance * .75) & (passes.end_x > passes.x)
    final_third = completed & (passes.x < 80) & (passes.end_x >= 80)
    box_entries = completed & ~((passes.x >= 102) & passes.y.between(18, 62)) & \
        ((passes.end_x >= 102) & passes.end_y.between(18, 62))
    switches = completed & ((passes.end_y - passes.y).abs() >= 35)
    forward = completed & (passes.end_x > passes.x + 1)
    return {
        "passes attempted": len(passes),
        "pass completion %": float(completed.mean() * 100) if len(passes) else 0,
        "forward passes": int(forward.sum()),
        "progressive passes": int(progressive.sum()),
        "final-third entries": int(final_third.sum()),
        "penalty-area entries": int(box_entries.sum()),
        "switches of play": int(switches.sum()),
    }


def team_summary(events: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    shots = events[series_or_default(events, "type").eq("Shot")].copy()
    passes = events[series_or_default(events, "type").eq("Pass")].copy()
    possession = possession_estimate(events, teams)
    rows = []
    for team in teams:
        team_shots = shots[series_or_default(shots, "team").eq(team)]
        team_passes = passes[series_or_default(passes, "team").eq(team)]
        xg = pd.to_numeric(series_or_default(team_shots, "shot_statsbomb_xg", 0), errors="coerce").sum()
        on_target = series_or_default(team_shots, "shot_outcome").isin(["Goal", "Saved", "Saved to Post"]).sum()
        rows.append({
            "team": team,
            "possession estimate %": round(float(possession.get(team, 0)), 1),
            "shots": len(team_shots),
            "shots on target": int(on_target),
            "goals": int(series_or_default(team_shots, "shot_outcome").eq("Goal").sum()),
            "xG": round(float(xg), 2),
            "passes attempted": len(team_passes),
            "pass completion %": round(float(series_or_default(team_passes, "pass_outcome").isna().mean() * 100), 1)
            if len(team_passes) else 0,
        })
    return pd.DataFrame(rows).set_index("team")


# -----------------------------------------------------------------------------
# Line-ups and player statistics
# -----------------------------------------------------------------------------
def timestamp_seconds(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    parts = str(value).split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
    except (TypeError, ValueError):
        return None
    return None


def lineup_table(lineups: dict, team: str, match_minutes: int) -> pd.DataFrame:
    rows = []
    for _, player in lineups.get(team, pd.DataFrame()).iterrows():
        periods = player.get("positions") or []
        if not periods:
            rows.append({"number": player.get("jersey_number"), "player": player.get("player_name"),
                         "position": "—", "status": "Unused substitute", "minutes": 0})
            continue
        starter = any(position.get("start_reason") == "Starting XI" for position in periods)
        total_seconds = 0.0
        for position in periods:
            start = timestamp_seconds(position.get("from")) or 0
            end = timestamp_seconds(position.get("to"))
            total_seconds += max(0, (end if end is not None else match_minutes * 60) - start)
        rows.append({
            "number": player.get("jersey_number"),
            "player": player.get("player_name"),
            "position": periods[0].get("position", "—"),
            "status": "Starter" if starter else "Substitute",
            "minutes": int(round(total_seconds / 60)),
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    order = {"Starter": 0, "Substitute": 1, "Unused substitute": 2}
    return result.assign(_order=result["status"].map(order)).sort_values(["_order", "number"]).drop(columns="_order")


def player_summary(events: pd.DataFrame, team: str) -> pd.DataFrame:
    team_events = events[series_or_default(events, "team").eq(team)]
    players = sorted(series_or_default(team_events, "player").dropna().unique())
    rows = []
    for player in players:
        pe = team_events[series_or_default(team_events, "player").eq(player)]
        passes = pe[series_or_default(pe, "type").eq("Pass")]
        shots = pe[series_or_default(pe, "type").eq("Shot")]
        completed_count = int(series_or_default(passes, "pass_outcome").isna().sum())
        rows.append({
            "player": player,
            "touch events": len(pe),
            "passes": len(passes),
            "completed passes": completed_count,
            "pass completion %": round(completed_count / len(passes) * 100, 1) if len(passes) else 0.0,
            "shots": len(shots),
            "xG": round(float(pd.to_numeric(series_or_default(shots, "shot_statsbomb_xg", 0), errors="coerce").sum()), 2),
            "pressures": int(series_or_default(pe, "type").eq("Pressure").sum()),
            "recoveries": int(series_or_default(pe, "type").eq("Ball Recovery").sum()),
            "interceptions": int(series_or_default(pe, "type").eq("Interception").sum()),
        })
    return pd.DataFrame(rows).sort_values("touch events", ascending=False) if rows else pd.DataFrame()


# -----------------------------------------------------------------------------
# Charts
# -----------------------------------------------------------------------------
def shot_map(shots: pd.DataFrame, teams: list[str]):
    shots = add_xy(shots).dropna(subset=["x", "y"])
    fig, ax = PITCH.draw(figsize=(12, 7))
    colours = [TEAM_COLOUR_MAP.get(team, fallback) for team, fallback in zip(teams, ["#d71920", "#111111"])]
    for team, colour in zip(teams, colours):
        team_shots = shots[series_or_default(shots, "team").eq(team)]
        xg = pd.to_numeric(series_or_default(team_shots, "shot_statsbomb_xg", 0.05), errors="coerce").fillna(0.05)
        goals = series_or_default(team_shots, "shot_outcome").eq("Goal")
        PITCH.scatter(team_shots.loc[~goals, "x"], team_shots.loc[~goals, "y"], s=80 + xg[~goals] * 550,
                      color=colour, edgecolors="white", alpha=.65, ax=ax, label=f"{team} — shot")
        PITCH.scatter(team_shots.loc[goals, "x"], team_shots.loc[goals, "y"], s=170 + xg[goals] * 600,
                      marker="*", color=colour, edgecolors="#ffd700", linewidth=1.5, ax=ax, label=f"{team} — goal")
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.015), ncol=2, frameon=False,
              columnspacing=2.5, handletextpad=.7, fontsize=9)
    ax.set_title("Shot map", color="white", pad=52, fontsize=15, weight="bold")
    fig.text(.5, .955, "Circle = shot  ·  Star = goal  ·  Marker size = xG",
             ha="center", va="center", color="#333333", fontsize=9)
    add_direction_label(fig)
    return fig


def xg_timeline(shots: pd.DataFrame, teams: list[str]):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    colours = [TEAM_COLOUR_MAP.get(team, fallback) for team, fallback in zip(teams, ["#d71920", "#111111"])]
    for team, colour in zip(teams, colours):
        team_shots = shots[series_or_default(shots, "team").eq(team)].sort_values("minute")
        minutes = pd.to_numeric(series_or_default(team_shots, "minute", 0), errors="coerce").fillna(0)
        xg = pd.to_numeric(series_or_default(team_shots, "shot_statsbomb_xg", 0), errors="coerce").fillna(0).cumsum()
        ax.step([0] + minutes.tolist(), [0] + xg.tolist(), where="post", color=colour, linewidth=2.5, label=team)
    ax.set(xlabel="Minute", ylabel="Cumulative xG", title="xG timeline")
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def average_positions(events: pd.DataFrame, team: str, lineups: dict):
    passes = add_xy(completed_passes(events, team)).dropna(subset=["x", "y", "player"])
    positions = passes.groupby("player").agg(x=("x", "mean"), y=("y", "mean"), actions=("id", "count"))
    numbers = lineups.get(team, pd.DataFrame())
    if not numbers.empty:
        positions["number"] = positions.index.map(numbers.set_index("player_name")["jersey_number"])
    else:
        positions["number"] = np.nan
    fig, ax = PITCH.draw(figsize=(10, 7))
    PITCH.scatter(positions.x, positions.y, s=np.clip(positions.actions * 12, 250, 750),
                  color=TEAM_COLOUR_MAP.get(team, "#d71920"), edgecolors="white", ax=ax)
    for player, row in positions.iterrows():
        label = str(int(row.number)) if pd.notna(row.number) else str(player).split()[-1][:3]
        ax.annotate(label, (row.x, row.y), ha="center", va="center", color="white", weight="bold")
    ax.set_title(f"{team} · average completed-pass locations", color="white")
    add_direction_label(fig)
    return fig


def pass_network(events: pd.DataFrame, team: str, lineups: dict, minimum: int):
    passes = add_xy(completed_passes(events, team)).dropna(subset=["x", "y", "player", "pass_recipient"])
    positions = passes.groupby("player").agg(x=("x", "mean"), y=("y", "mean"), touches=("id", "count"))
    pairs = passes.assign(pair=passes.apply(lambda r: tuple(sorted((r["player"], r["pass_recipient"]))), axis=1))
    pairs = pairs.groupby("pair").size().reset_index(name="passes")
    pairs = pairs[pairs.passes >= minimum]
    numbers = lineups.get(team, pd.DataFrame())
    number_map = numbers.set_index("player_name")["jersey_number"] if not numbers.empty else pd.Series(dtype=float)
    fig, ax = PITCH.draw(figsize=(12, 8))
    for _, row in pairs.iterrows():
        p1, p2 = row.pair
        if p1 in positions.index and p2 in positions.index:
            ax.plot([positions.at[p1, "x"], positions.at[p2, "x"]],
                    [positions.at[p1, "y"], positions.at[p2, "y"]],
                    color="#f2d338", alpha=.25 + min(row.passes / 25, .6),
                    linewidth=min(.6 + row.passes / 4, 6), zorder=1)
    PITCH.scatter(positions.x, positions.y, s=np.clip(positions.touches * 10, 250, 700),
                  color=TEAM_COLOUR_MAP.get(team, "#d71920"), edgecolors="white", ax=ax, zorder=2)
    for player, row in positions.iterrows():
        number = number_map.get(player, np.nan)
        label = str(int(number)) if pd.notna(number) else str(player).split()[-1][:3]
        ax.annotate(label, (row.x, row.y), ha="center", va="center", color="white", weight="bold", zorder=3)
    ax.set_title(f"{team} · completed-pass network (minimum {minimum})", color="white")
    add_direction_label(fig)
    return fig


def action_map(events: pd.DataFrame, team: str, event_types: Iterable[str], title: str):
    selected = events[series_or_default(events, "team").eq(team) & series_or_default(events, "type").isin(event_types)]
    selected = add_xy(selected).dropna(subset=["x", "y"])
    fig, ax = PITCH.draw(figsize=(10, 7))
    palette = {
        "Pressure": ("#f2d338", "o"),
        "Ball Recovery": ("#42a5f5", "o"),
        "Interception": ("#ff7043", "s"),
        "Block": ("#ab47bc", "D"),
        "Duel": ("#ffffff", "^"),
    }
    for event_type in event_types:
        group = selected[series_or_default(selected, "type").eq(event_type)]
        if group.empty:
            continue
        colour, marker = palette.get(event_type, ("#f2d338", "o"))
        PITCH.scatter(group.x, group.y, s=58, color=colour, marker=marker,
                      edgecolors="#111111", linewidth=.7, alpha=.78, ax=ax,
                      label=f"{event_type} ({len(group)})")
    if not selected.empty:
        ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=2,
                  frameon=False, fontsize=9, columnspacing=1.5)
    ax.set_title(title, color="white", pad=45, fontsize=14, weight="bold")
    add_direction_label(fig)
    return fig


def channel_entries_chart(events: pd.DataFrame, team: str):
    passes = completed_passes(events, team)
    passes = add_xy(passes, "pass_end_location", "end_").dropna(subset=["end_x", "end_y"])
    entries = passes[(passes.end_x >= 80) & (passes.end_x > add_xy(passes).x)]
    labels = ["Left", "Centre", "Right"]
    counts = [int((entries.end_y < 26.7).sum()), int(entries.end_y.between(26.7, 53.3).sum()),
              int((entries.end_y > 53.3).sum())]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, counts, color=["#b54749", "#526d82", "#2b7a3d"])
    ax.bar_label(bars)
    ax.set(title=f"{team} · attacking-third entries by channel", ylabel="Completed entries")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def number_reference(lineups: dict, team: str, active_players: Iterable[str]) -> pd.DataFrame:
    lineup = lineups.get(team, pd.DataFrame()).copy()
    if lineup.empty:
        return pd.DataFrame(columns=["number", "player"])
    result = lineup[lineup.player_name.isin(set(active_players))][["jersey_number", "player_name"]]
    return result.rename(columns={"jersey_number": "number", "player_name": "player"}).sort_values("number")


# -----------------------------------------------------------------------------
# Transparent coaching observations
# -----------------------------------------------------------------------------
def coaching_observations(events: pd.DataFrame, team: str, opponent: str) -> list[str]:
    notes = []
    shots = events[series_or_default(events, "type").eq("Shot")]
    team_shots = shots[series_or_default(shots, "team").eq(team)]
    opp_shots = shots[series_or_default(shots, "team").eq(opponent)]
    team_xg = pd.to_numeric(series_or_default(team_shots, "shot_statsbomb_xg", 0), errors="coerce").sum()
    opp_xg = pd.to_numeric(series_or_default(opp_shots, "shot_statsbomb_xg", 0), errors="coerce").sum()
    passes = add_xy(add_xy(completed_passes(events, team), "pass_end_location", "end_"))
    metrics = pass_metrics(events, team)
    located = passes.dropna(subset=["x", "y"])
    territorial_share = float((located.x >= 60).mean() * 100) if len(located) else 0
    wide_share = float((~located.y.between(26.7, 53.3)).mean() * 100) if len(located) else 0
    entries = passes[(passes.end_x >= 80) & (passes.x < 80)]
    left_entries = int((entries.end_y < 26.7).sum())
    right_entries = int((entries.end_y > 53.3).sum())
    pressures = count_type(events, "Pressure", team)
    high_recoveries = add_xy(events[series_or_default(events, "team").eq(team) &
                                    series_or_default(events, "type").eq("Ball Recovery")])
    high_recovery_count = int((high_recoveries.x >= 80).sum())

    if team_xg >= opp_xg + .75:
        notes.append(f"Chance quality favoured {team}: {team_xg:.2f} xG versus {opp_xg:.2f}.")
    elif opp_xg >= team_xg + .75:
        notes.append(f"The opponent created the stronger chances: {opp_xg:.2f} xG versus {team_xg:.2f}.")
    if len(team_shots) >= 10 and team_xg / max(len(team_shots), 1) < .08:
        notes.append(f"Shot selection may need review: {len(team_shots)} shots averaged only {team_xg / len(team_shots):.2f} xG each.")
    notes.append(f"Territorial profile: {territorial_share:.1f}% of completed-pass origins were in the attacking half; {wide_share:.1f}% were in wide channels.")
    notes.append(f"Progression: {metrics['progressive passes']} progressive passes, {metrics['final-third entries']} final-third entries and {metrics['penalty-area entries']} penalty-area entries.")
    if left_entries >= max(3, right_entries * 1.5):
        notes.append(f"Attacking entries favoured the left channel ({left_entries} versus {right_entries} on the right).")
    elif right_entries >= max(3, left_entries * 1.5):
        notes.append(f"Attacking entries favoured the right channel ({right_entries} versus {left_entries} on the left).")
    if pressures:
        notes.append(f"StatsBomb recorded {pressures} pressure events and {high_recovery_count} attacking-third recoveries.")
    return notes or ["The selected period is too limited for a strong rule-based observation."]


def comparative_coaching_summary(events: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    summary = team_summary(events, teams).reset_index()
    summary["xG per shot"] = (summary["xG"] / summary["shots"].replace(0, np.nan)).fillna(0).round(2)
    summary["pressures"] = summary.team.map(lambda team: count_type(events, "Pressure", team))
    summary["recoveries"] = summary.team.map(lambda team: count_type(events, "Ball Recovery", team))
    for metric in ["progressive passes", "final-third entries", "penalty-area entries"]:
        summary[metric] = summary.team.map(lambda team: pass_metrics(events, team)[metric])
    return summary


def generated_match_analysis(events: pd.DataFrame, team: str, opponent: str,
                             length: str = "Standard", focus: str = "Complete") -> dict[str, str]:
    teams = [team, opponent]
    summary = team_summary(events, teams)
    own, other = summary.loc[team], summary.loc[opponent]
    own_pass, other_pass = pass_metrics(events, team), pass_metrics(events, opponent)
    own_shots = events[series_or_default(events, "type").eq("Shot") & series_or_default(events, "team").eq(team)]
    other_shots = events[series_or_default(events, "type").eq("Shot") & series_or_default(events, "team").eq(opponent)]
    own_xg_shot = float(own.xG / own.shots) if own.shots else 0
    other_xg_shot = float(other.xG / other.shots) if other.shots else 0
    passes = add_xy(add_xy(completed_passes(events, team), "pass_end_location", "end_")).dropna(subset=["x", "y"])
    entries = passes[(passes.x < 80) & (passes.end_x >= 80)]
    left = int((entries.end_y < 26.7).sum())
    centre = int(entries.end_y.between(26.7, 53.3).sum())
    right = int((entries.end_y > 53.3).sum())
    channel_counts = {"left": left, "central": centre, "right": right}
    main_channel = max(channel_counts, key=channel_counts.get) if entries.shape[0] else "no clear"
    own_pressures, other_pressures = count_type(events, "Pressure", team), count_type(events, "Pressure", opponent)
    own_recoveries, other_recoveries = count_type(events, "Ball Recovery", team), count_type(events, "Ball Recovery", opponent)
    high_recoveries = add_xy(events[series_or_default(events, "team").eq(team) &
                                    series_or_default(events, "type").eq("Ball Recovery")])
    high_recovery_count = int((high_recoveries.x >= 80).sum())
    player_stats = player_summary(events, team)

    if own.xG > other.xG + .4:
        balance = f"{team} created the stronger chances"
    elif other.xG > own.xG + .4:
        balance = f"{opponent} created the stronger chances"
    else:
        balance = "chance quality was relatively even"
    sections = {
        "Match summary": (
            f"{balance}. {team} recorded {int(own.shots)} shots and {own.xG:.2f} xG, compared with "
            f"{int(other.shots)} shots and {other.xG:.2f} xG for {opponent}. The possession estimate was "
            f"{own['possession estimate %']:.1f}% for {team}."
        ),
        "In possession": (
            f"{team} completed {own_pass['pass completion %']:.1f}% of attempted passes, including "
            f"{own_pass['progressive passes']} progressive passes, {own_pass['final-third entries']} final-third "
            f"entries and {own_pass['penalty-area entries']} penalty-area entries. The {main_channel} channel was "
            f"used most often for recorded final-third entries ({left} left, {centre} central and {right} right)."
        ),
        "Out of possession": (
            f"The data records {own_pressures} pressures, {own_recoveries} ball recoveries and "
            f"{count_type(events, 'Interception', team)} interceptions for {team}. {high_recovery_count} recoveries "
            f"occurred in the attacking third. {opponent} recorded {other_pressures} pressures and "
            f"{other_recoveries} recoveries over the same period."
        ),
        "Transitions": (
            f"{team}'s progression produced {own_pass['final-third entries']} completed entries into the final third "
            f"and {own_pass['penalty-area entries']} entries into the penalty area. The {high_recovery_count} "
            f"attacking-third recoveries provide the clearest event-data indicator of opportunities to attack "
            f"immediately after regaining possession."
        ),
    }

    first_half = events[pd.to_numeric(series_or_default(events, "minute", 0), errors="coerce") <= 45]
    second_half = events[pd.to_numeric(series_or_default(events, "minute", 0), errors="coerce") > 45]
    first = team_summary(first_half, teams).loc[team]
    second = team_summary(second_half, teams).loc[team]
    sections["Key periods"] = (
        f"{team} produced {int(first.shots)} shots and {first.xG:.2f} xG in the first half, followed by "
        f"{int(second.shots)} shots and {second.xG:.2f} xG in the second half. Pass completion changed from "
        f"{first['pass completion %']:.1f}% to {second['pass completion %']:.1f}%."
    )
    if not player_stats.empty:
        involved = player_stats.iloc[0]
        shooter = player_stats.sort_values("xG", ascending=False).iloc[0]
        defender = player_stats.assign(defensive=player_stats.pressures + player_stats.recoveries +
                                       player_stats.interceptions).sort_values("defensive", ascending=False).iloc[0]
        sections["Player observations"] = (
            f"{involved.player} had the highest recorded event involvement ({int(involved['touch events'])} events). "
            f"{shooter.player} generated the highest individual xG ({shooter.xG:.2f}), while {defender.player} "
            f"recorded the most combined pressures, recoveries and interceptions ({int(defender.defensive)})."
        )
    strengths = []
    priorities = []
    training = []
    if own.xG >= other.xG:
        strengths.append(f"chance creation ({own.xG:.2f} xG versus {other.xG:.2f})")
    else:
        priorities.append(f"restricting opponent chance quality ({other.xG:.2f} xG conceded)")
    if own_pass["final-third entries"] > other_pass["final-third entries"]:
        strengths.append(f"progression into the final third ({own_pass['final-third entries']} entries)")
    else:
        priorities.append(f"progression into the final third ({own_pass['final-third entries']} entries)")
    if own_xg_shot < .09 and len(own_shots) >= 5:
        priorities.append(f"shot selection ({own_xg_shot:.2f} xG per shot)")
        training.append("use conditioned finishing practices that reward central shots and an extra pass before low-quality attempts")
    if own_pass["penalty-area entries"] < max(4, own_pass["final-third entries"] * .3):
        priorities.append("turning final-third possession into penalty-area entries")
        training.append("rehearse wide and central combinations that finish with multiple players occupying the penalty area")
    if high_recovery_count >= 4:
        strengths.append(f"attacking-third recoveries ({high_recovery_count})")
    else:
        training.append("train the first five seconds after possession is lost, with clear pressure, cover and recovery roles")
    sections["Strengths"] = "The clearest data-supported strengths were " + "; ".join(strengths[:3]) + "." if strengths else "No single strength met the report thresholds in the selected period."
    sections["Development priorities"] = "Priority areas are " + "; ".join(priorities[:3]) + "." if priorities else "No major development priority met the report thresholds in the selected period."
    sections["Training recommendations"] = "Recommended next steps are to " + "; and ".join(training[:3]) + "." if training else "Maintain the current themes and use video to confirm the off-ball detail behind the positive indicators."

    focus_sections = {
        "In possession": {"Match summary", "In possession", "Transitions", "Player observations", "Strengths", "Development priorities", "Training recommendations"},
        "Out of possession": {"Match summary", "Out of possession", "Transitions", "Player observations", "Strengths", "Development priorities", "Training recommendations"},
        "Transitions": {"Match summary", "Transitions", "Key periods", "Strengths", "Development priorities", "Training recommendations"},
    }
    if focus != "Complete":
        sections = {key: value for key, value in sections.items() if key in focus_sections[focus]}
    if length == "Brief":
        keep = ["Match summary", "Strengths", "Development priorities", "Training recommendations"]
        sections = {key: sections[key] for key in keep if key in sections}
    elif length == "Detailed":
        sections["Data context"] = (
            f"The selected sample contains {len(events)} events and {len(own_shots) + len(other_shots)} shots. "
            f"It includes {own_pass['passes attempted']} pass attempts by {team} and "
            f"{other_pass['passes attempted']} by {opponent}."
        )
    return sections


def make_pdf_report(match_title: str, period: tuple[int, int], summary: pd.DataFrame,
                    team_notes: dict[str, list[str]], coach_fields: dict,
                    figures: list[tuple[str, object]]) -> bytes:
    output = BytesIO()
    with PdfPages(output) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        fig.text(.07, .95, "COACHING MATCH REPORT", fontsize=20, weight="bold", color="#b54749")
        fig.text(.07, .915, match_title, fontsize=14, weight="bold")
        fig.text(.07, .89, f"Analysis period: {period[0]}–{period[1]} minutes", fontsize=10)
        y = .85
        for heading, key in [("Match plan", "match_plan"), ("Strengths", "strengths"),
                             ("Development priorities", "priorities"), ("Video evidence", "video"),
                             ("Training response", "training")]:
            fig.text(.07, y, heading, fontsize=12, weight="bold", color="#263746")
            y -= .025
            content = coach_fields.get(key, "").strip() or "—"
            wrapped = textwrap.wrap(content, 100)[:5]
            fig.text(.08, y, "\n".join(wrapped), fontsize=9, va="top")
            y -= max(.07, .018 * len(wrapped) + .035)
        fig.text(.07, y, "Data summary", fontsize=12, weight="bold", color="#263746")
        y -= .03
        for _, row in summary.iterrows():
            line = (f"{row['team']}: {row['shots']} shots, {row['xG']:.2f} xG, "
                    f"{row['pass completion %']:.1f}% passing, {row.get('progressive passes', 0)} progressive passes")
            fig.text(.08, y, line, fontsize=9)
            y -= .022
        y -= .02
        fig.text(.07, y, "Data-based observations", fontsize=12, weight="bold", color="#263746")
        y -= .03
        for team, notes in team_notes.items():
            fig.text(.08, y, team, fontsize=10, weight="bold")
            y -= .022
            for note in notes[:4]:
                wrapped = textwrap.wrap("• " + note, 105)
                fig.text(.09, y, "\n".join(wrapped), fontsize=8.5, va="top")
                y -= .018 * len(wrapped) + .008
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        generated_sections = coach_fields.get("generated_sections", {})
        section_items = list(generated_sections.items())
        for page_start in range(0, len(section_items), 4):
            analysis_page = plt.figure(figsize=(8.27, 11.69))
            analysis_page.patch.set_facecolor("white")
            analysis_page.text(.07, .95, "GENERATED MATCH ANALYSIS", fontsize=18, weight="bold", color="#b54749")
            analysis_page.text(.07, .92, match_title, fontsize=11, weight="bold")
            y = .87
            for heading, content in section_items[page_start:page_start + 4]:
                analysis_page.text(.07, y, heading, fontsize=12, weight="bold", color="#263746")
                y -= .03
                wrapped = textwrap.wrap(str(content), 100)
                analysis_page.text(.08, y, "\n".join(wrapped), fontsize=9, va="top", linespacing=1.4)
                y -= max(.12, .021 * len(wrapped) + .065)
            pdf.savefig(analysis_page, bbox_inches="tight")
            plt.close(analysis_page)
        for title, chart in figures:
            chart.suptitle(title, fontsize=14, weight="bold")
            pdf.savefig(chart, bbox_inches="tight")
            plt.close(chart)
    output.seek(0)
    return output.getvalue()


# -----------------------------------------------------------------------------
# Interface
# -----------------------------------------------------------------------------
st.title("⚽ Football Coaching Report")
st.caption("Analyse StatsBomb open data or upload event data from another source")

with st.sidebar:
    st.header("Data source")
    source_mode = st.radio("Choose source", ["StatsBomb open data", "Upload match data"])

source_is_statsbomb = source_mode == "StatsBomb open data"

if source_is_statsbomb:
    try:
        competitions = load_competitions()
    except Exception as exc:
        st.error(f"StatsBomb competitions could not be loaded: {exc}")
        st.stop()
    with st.sidebar:
        st.header("Match selection")
        competition_name = st.selectbox("Competition", sorted(competitions.competition_name.unique()))
        competition_rows = competitions[competitions.competition_name.eq(competition_name)]
        season_name = st.selectbox("Season", sorted(competition_rows.season_name.unique(), reverse=True))
        selected_row = competition_rows[competition_rows.season_name.eq(season_name)].iloc[0]
        try:
            matches = load_matches(int(selected_row.competition_id), int(selected_row.season_id)).copy()
        except Exception as exc:
            st.error(f"Matches could not be loaded: {exc}")
            st.stop()
        matches["label"] = matches.home_team + " vs " + matches.away_team + " · " + matches.match_date.astype(str)
        selected_label = st.selectbox("Match", matches.label.tolist())
        chosen_match = matches[matches.label.eq(selected_label)].iloc[0]
        if st.button("Generate report", type="primary", use_container_width=True):
            st.session_state.report_match_id = int(chosen_match.match_id)
    if "report_match_id" not in st.session_state:
        st.info("Choose a competition, season and match, then select **Generate report**.")
        st.stop()
    match_id = st.session_state.report_match_id
    stored = matches[matches.match_id.eq(match_id)]
    if stored.empty:
        st.warning("The selected competition or season changed. Generate the report again.")
        st.stop()
    match = stored.iloc[0]
    try:
        with st.spinner("Loading match events and line-ups…"):
            all_events, lineups = load_match(match_id)
    except Exception as exc:
        st.error(f"This match could not be loaded: {exc}")
        st.stop()
    home_colour_default, away_colour_default = "#D71920", "#1F2933"
else:
    st.subheader("Upload match event data")
    t1, t2 = st.columns(2)
    with t1:
        st.download_button("Download team event template", custom_template().to_csv(index=False).encode("utf-8"),
                           "football_event_template.csv", "text/csv")
    with t2:
        st.download_button("Download template field guide", template_dictionary().to_csv(index=False).encode("utf-8"),
                           "football_event_template_field_guide.csv", "text/csv")
    st.caption("The template contains three examples. Delete those rows before recording a real match.")
    uploaded = st.file_uploader("Upload CSV, TSV, Excel, JSON or XML", type=["csv", "tsv", "xlsx", "xls", "json", "xml"])
    if uploaded is None:
        st.info("Upload an event file or download the template to begin.")
        st.stop()
    try:
        raw_upload = read_uploaded_file(uploaded)
    except Exception as exc:
        st.error(f"The file could not be read: {exc}")
        st.stop()
    if raw_upload.empty:
        st.error("The uploaded file contains no rows.")
        st.stop()
    st.markdown("#### Upload preview")
    st.table(format_table(raw_upload.head(8), hide_index=True))
    guessed = guess_mapping(raw_upload.columns)
    st.markdown("#### Match data setup")
    options = ["— Not supplied —"] + list(raw_upload.columns)
    mapping = {}
    automatically_mapped = sum(value is not None for value in guessed.values())
    required_unmapped = [field for field in ["minute", "team", "event_type"] if not guessed.get(field)]
    if required_unmapped:
        st.warning("Some essential fields need your attention: " + ", ".join(required_unmapped))
    else:
        st.success(f"File recognised. {automatically_mapped} fields matched automatically.")
    with st.expander("Review or change field mapping", expanded=bool(required_unmapped)):
        st.caption("Only change these selections if a column has been matched incorrectly.")
        map_columns = st.columns(3)
        for index, field in enumerate(TEMPLATE_COLUMNS):
            default_column = guessed.get(field)
            default_index = options.index(default_column) if default_column in options else 0
            selected = map_columns[index % 3].selectbox(field.replace("_", " ").title(), options,
                                                        index=default_index, key=f"map_{field}")
            mapping[field] = None if selected.startswith("—") else selected
    missing = [field for field in ["minute", "team", "event_type"] if not mapping.get(field)]
    if missing:
        st.warning("The file can be displayed, but event analysis needs mappings for: " + ", ".join(missing))
        st.table(format_table(raw_upload, hide_index=True))
        st.download_button("Download displayed data", raw_upload.to_csv(index=False).encode("utf-8"),
                           "uploaded_data.csv", "text/csv")
        st.stop()
    raw_event_values = column_values(raw_upload, mapping, "event_type").dropna().astype(str).str.strip().unique().tolist()
    event_value_map = {}
    recognised_event_count = sum(canonical_event_name(value) in CANONICAL_EVENTS and
                                 canonical_event_name(value) != "Unknown" for value in raw_event_values)
    if recognised_event_count == len(raw_event_values):
        st.success(f"All {len(raw_event_values)} event labels recognised.")
    else:
        st.warning(f"{len(raw_event_values) - recognised_event_count} event labels need checking.")
    with st.expander("Review or change event names", expanded=recognised_event_count != len(raw_event_values)):
        st.caption("Only change these selections if your software uses a custom tag or an event was interpreted incorrectly.")
        if len(raw_event_values) > 60:
            st.warning("Only the first 60 distinct event labels are shown. Consolidate highly variable labels before upload.")
        event_columns = st.columns(3)
        for index, raw_value in enumerate(raw_event_values[:60]):
            guessed_event = canonical_event_name(raw_value)
            event_options = CANONICAL_EVENTS + ([guessed_event] if guessed_event not in CANONICAL_EVENTS else [])
            selected_event = event_columns[index % 3].selectbox(
                raw_value, event_options, index=event_options.index(guessed_event), key=f"event_map_{index}_{raw_value}")
            event_value_map[raw_value] = selected_event
    team_values = column_values(raw_upload, mapping, "team").dropna().astype(str).str.strip().unique().tolist()
    home_default = team_values[0] if team_values else "Home Team"
    mapped_away = column_values(raw_upload, mapping, "away_team").dropna().astype(str).str.strip().unique().tolist()
    away_default = team_values[1] if len(team_values) > 1 else (mapped_away[0] if mapped_away else "Away Team")
    m1, m2, m3 = st.columns(3)
    home_team = m1.text_input("Home team", home_default)
    away_team = m2.text_input("Away team", away_default)
    match_date = m3.text_input("Match date", str(column_values(raw_upload, mapping, "match_date", "Unknown").iloc[0]))
    c1, c2, c3 = st.columns(3)
    coordinate_system = c1.selectbox("Coordinate system", ["120×80", "0–100", "0–1"])
    time_unit = c2.selectbox("Time field unit", ["Minutes", "Seconds"])
    orientation = c3.selectbox("Coordinate orientation", ["Each team attacks left to right", "Single fixed pitch direction"])
    mapping_for_import = dict(mapping)
    import_raw = raw_upload.copy()
    if time_unit == "Seconds":
        minute_source = mapping.get("minute")
        import_raw["_converted_minute"] = pd.to_numeric(import_raw[minute_source], errors="coerce") / 60
        mapping_for_import["minute"] = "_converted_minute"
    try:
        all_events = normalise_uploaded_events(import_raw, mapping_for_import, coordinate_system,
                                                home_team, away_team, orientation == "Single fixed pitch direction",
                                                event_value_map)
    except Exception as exc:
        st.error(f"The mapped data could not be normalised: {exc}")
        st.stop()
    lineups = uploaded_lineups(raw_upload, mapping, [home_team, away_team])
    home_score = int((all_events.team.eq(home_team) & all_events.type.eq("Shot") &
                      all_events.shot_outcome.eq("Goal")).sum())
    away_score = int((all_events.team.eq(away_team) & all_events.type.eq("Shot") &
                      all_events.shot_outcome.eq("Goal")).sum())
    match_id = f"upload_{uploaded.name.rsplit('.', 1)[0]}"
    selected_label = f"{home_team} vs {away_team} · {match_date}"
    match = pd.Series({"match_id": match_id, "match_date": match_date, "home_team": home_team,
                       "away_team": away_team, "home_score": home_score, "away_score": away_score})
    matches = pd.DataFrame([{**match.to_dict(), "label": selected_label}])
    home_colour_default = valid_hex_colour(column_values(raw_upload, mapping, "home_colour", "#D71920").iloc[0], "#D71920")
    away_colour_default = valid_hex_colour(column_values(raw_upload, mapping, "away_colour", "#1F2933").iloc[0], "#1F2933")
    with st.expander("Import validation"):
        valid_locations = int(all_events.location.notna().sum())
        st.write(f"Rows imported: **{len(all_events)}**")
        st.write(f"Rows with pitch coordinates: **{valid_locations}**")
        st.write(f"Recognised event types: **{all_events.type.nunique()}**")
        unknown = int(all_events.type.eq("Unknown").sum())
        if unknown:
            st.warning(f"{unknown} rows have no recognised event type.")

home_team, away_team = match.home_team, match.away_team
teams = [home_team, away_team]
end_minute = match_end_minute(all_events)

with st.sidebar:
    st.divider()
    st.header("Analysis filters")
    with st.expander("Team colours", expanded=False):
        home_colour = st.color_picker(home_team, home_colour_default, key=f"colour_home_{home_team}")
        away_colour = st.color_picker(away_team, away_colour_default, key=f"colour_away_{away_team}")
    TEAM_COLOUR_MAP.clear()
    TEAM_COLOUR_MAP.update({home_team: home_colour, away_team: away_colour})
    focus_team = st.radio("Team to analyse", teams)
    first_goal_values = pd.to_numeric(all_events[
        series_or_default(all_events, "type").eq("Shot") &
        series_or_default(all_events, "shot_outcome").eq("Goal")
    ].get("minute", pd.Series(dtype=float)), errors="coerce").dropna()
    first_sub_values = pd.to_numeric(all_events[
        series_or_default(all_events, "type").eq("Substitution")
    ].get("minute", pd.Series(dtype=float)), errors="coerce").dropna()
    first_goal = int(first_goal_values.min()) if not first_goal_values.empty else None
    first_sub = int(first_sub_values.min()) if not first_sub_values.empty else None
    period_options = {
        "Full match": (0, end_minute),
        "First half": (0, min(45, end_minute)),
        "Second half": (min(46, end_minute), end_minute),
    }
    if first_goal is not None:
        period_options["Before first goal"] = (0, first_goal)
        period_options["After first goal"] = (first_goal, end_minute)
    if first_sub is not None:
        period_options["Before first substitution"] = (0, first_sub)
        period_options["After first substitution"] = (first_sub, end_minute)
    period_options["Custom"] = (0, end_minute)
    period_choice = st.selectbox("Quick period", list(period_options))
    if period_choice == "Custom":
        minute_range = st.slider("Minute range", 0, end_minute, (0, end_minute))
    else:
        minute_range = period_options[period_choice]
        st.caption(f"Minutes {minute_range[0]}–{minute_range[1]}")
    match_state = st.selectbox("Match state", ["All states", "Drawing", "Winning", "Losing"])
    minimum_connection = st.slider("Minimum passes in network", 2, 15, 5)
    st.info(f"Detailed tabs currently show **{focus_team}**")

period_events = all_events[
    pd.to_numeric(series_or_default(all_events, "minute", 0), errors="coerce").between(*minute_range)
].copy()
state_selection = match_state_mask(all_events, focus_team, match_state).reindex(period_events.index).fillna(False)
events = period_events[state_selection].copy()
opponent = away_team if focus_team == home_team else home_team

st.header(f"{home_team} {int(match.home_score)}–{int(match.away_score)} {away_team}")
st.caption(f"{match.match_date} · minutes {minute_range[0]}–{minute_range[1]} · {match_state.lower()}")

st.markdown(
    f"""
    <div style="background:#e9f0f4;color:#18232b;padding:12px 16px;border-left:7px solid #b54749;
                border-radius:5px;margin:8px 0 18px 0;font-size:1.05rem;">
        <strong>Detailed analysis team:</strong> {focus_team}<br>
        <span style="font-size:.9rem;">Opponent: {opponent} · Period: {period_choice} · State: {match_state}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

summary = team_summary(events, teams)
focus = summary.loc[focus_team]
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Possession estimate", f"{focus['possession estimate %']:.1f}%")
m2.metric("Shots", int(focus.shots))
m3.metric("Shots on target", int(focus["shots on target"]))
m4.metric("xG", f"{focus.xG:.2f}")
m5.metric("Pass completion", f"{focus['pass completion %']:.1f}%")
st.caption("Possession is an event-duration estimate, not StatsBomb's official possession statistic.")

tabs = st.tabs(["Overview", "Attacking", "Possession", "Defending", "Players", "Timeline",
                "Opponent", "Match comparison", "Coaching summary", "Report builder", "Event data"])

with tabs[0]:
    st.subheader("Match overview")
    st.table(format_table(summary))
    st.download_button("Download overview CSV", csv_bytes(summary.reset_index()), "match_overview.csv", "text/csv")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(home_team)
        st.table(format_table(lineup_table(lineups, home_team, end_minute), hide_index=True))
    with c2:
        st.subheader(away_team)
        st.table(format_table(lineup_table(lineups, away_team, end_minute), hide_index=True))

with tabs[1]:
    shots = events[series_or_default(events, "type").eq("Shot")].copy()
    st.subheader("Shot locations and chance quality")
    st.caption("Both teams are shown so you can compare where chances were created. Larger markers indicate higher-quality chances.")
    draw_and_show(shot_map(shots, teams), f"{match_id}_shot_map.png")
    st.subheader("How expected goals developed")
    draw_and_show(xg_timeline(shots, teams), f"{match_id}_xg_timeline.png")
    st.subheader(f"Progression and territory · {focus_team}")
    attack_metrics = pd.DataFrame([{"team": focus_team, **pass_metrics(events, focus_team)}])
    st.table(format_table(attack_metrics, hide_index=True))
    draw_and_show(channel_entries_chart(events, focus_team), f"{match_id}_{focus_team}_channels.png")
    if not shots.empty:
        columns = [c for c in ["minute", "team", "player", "shot_outcome", "shot_statsbomb_xg", "shot_body_part", "shot_type"] if c in shots]
        st.table(format_table(shots[columns], hide_index=True))
        st.download_button("Download shots CSV", csv_bytes(shots[columns]), "shots.csv", "text/csv")

with tabs[2]:
    st.subheader(f"Possession structure · {focus_team}")
    st.caption("These diagrams use completed-pass origins. They show on-ball structure, not the players’ complete off-ball positions.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Average pass locations")
        st.caption("Each marker shows a player's average location when completing a pass. Marker size reflects completed passes.")
        draw_and_show(average_positions(events, focus_team, lineups), f"{match_id}_{focus_team}_average_positions.png")
    with c2:
        st.markdown("#### Passing connections")
        st.caption(f"Lines join players who exchanged at least {minimum_connection} completed passes. Thicker lines mean more passes.")
        draw_and_show(pass_network(events, focus_team, lineups, minimum_connection), f"{match_id}_{focus_team}_pass_network.png")
    passes = events[series_or_default(events, "type").eq("Pass") & series_or_default(events, "team").eq(focus_team)]
    active_players = series_or_default(passes, "player").dropna().unique()
    st.markdown("#### Shirt-number key")
    st.table(format_table(number_reference(lineups, focus_team, active_players), hide_index=True))
    st.download_button("Download team passes CSV", csv_bytes(passes), "team_passes.csv", "text/csv")

with tabs[3]:
    st.subheader(f"Defensive activity · {focus_team}")
    st.caption("The locations are where StatsBomb recorded each action. Use the colour-and-shape keys above the pitches.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Pressures")
        st.caption("Where a defender applied pressure to an opponent in possession.")
        draw_and_show(action_map(events, focus_team, ["Pressure"], "Pressure locations"),
                      f"{match_id}_{focus_team}_pressures.png")
    with c2:
        st.markdown("#### Defensive actions")
        st.caption("Recoveries, interceptions, blocks and duels are separated by colour and marker shape.")
        draw_and_show(action_map(events, focus_team,
                                 ["Ball Recovery", "Interception", "Block", "Duel"],
                                 "Defensive-action locations"), f"{match_id}_{focus_team}_defending.png")
    defensive_rows = []
    for team in teams:
        defensive_rows.append({"team": team, "pressures": count_type(events, "Pressure", team),
                               "recoveries": count_type(events, "Ball Recovery", team),
                               "interceptions": count_type(events, "Interception", team),
                               "duels": count_type(events, "Duel", team),
                               "fouls": count_type(events, "Foul Committed", team)})
    st.table(format_table(pd.DataFrame(defensive_rows).set_index("team")))

with tabs[4]:
    stats = player_summary(events, focus_team)
    st.table(format_table(stats, hide_index=True))
    st.download_button("Download player statistics CSV", csv_bytes(stats), "player_statistics.csv", "text/csv")
    if not stats.empty:
        selected_player = st.selectbox("Player action map", stats.player.tolist())
        player_events = events[series_or_default(events, "player").eq(selected_player)]
        draw_and_show(action_map(player_events, focus_team, player_events.type.dropna().unique(), f"{selected_player} events"),
                      f"{match_id}_{selected_player}_events.png")
        st.subheader("Player comparison")
        comparison_players = st.multiselect("Compare two players", stats.player.tolist(),
                                            default=stats.player.tolist()[:2], max_selections=2)
        if comparison_players:
            compared = stats[stats.player.isin(comparison_players)].set_index("player").T
            compared.index.name = "Statistic"
            st.table(format_table(compared))

with tabs[5]:
    goal_mask = series_or_default(events, "type").eq("Shot") & series_or_default(events, "shot_outcome").eq("Goal")
    goals = events[goal_mask]
    substitutions = events[series_or_default(events, "type").eq("Substitution")]
    cards = events[
        series_or_default(events, "foul_committed_card").notna() |
        series_or_default(events, "bad_behaviour_card").notna()
    ]
    for heading, frame in [("Goals", goals), ("Substitutions", substitutions), ("Cards", cards)]:
        st.subheader(heading)
        columns = [c for c in ["minute", "team", "player", "substitution_replacement",
                                "foul_committed_card", "bad_behaviour_card"] if c in frame]
        st.table(format_table(frame[columns] if columns else pd.DataFrame(), hide_index=True))

with tabs[6]:
    st.subheader(f"Opponent analysis · {opponent}")
    st.caption(f"This view presents {opponent}'s attacking structure and key indicators from {focus_team}'s perspective.")
    opponent_metrics = pd.DataFrame([{"team": opponent, **pass_metrics(events, opponent)}])
    st.table(format_table(opponent_metrics, hide_index=True))
    o1, o2 = st.columns(2)
    with o1:
        st.markdown("#### Opponent passing structure")
        draw_and_show(pass_network(events, opponent, lineups, minimum_connection),
                      f"{match_id}_{opponent}_pass_network.png")
    with o2:
        st.markdown("#### Opponent defensive activity")
        draw_and_show(action_map(events, opponent, ["Pressure", "Ball Recovery", "Interception"],
                                 "Opponent pressures and recoveries"), f"{match_id}_{opponent}_actions.png")
    st.markdown("#### Opponent tendencies")
    for note in coaching_observations(events, opponent, focus_team):
        st.info(note)
    st.markdown("#### Most involved opponent players")
    opponent_players = player_summary(events, opponent)
    st.table(format_table(opponent_players.head(8), hide_index=True))

with tabs[7]:
    st.subheader(f"Compare another {focus_team} match")
    candidates = matches[(matches.home_team.eq(focus_team) | matches.away_team.eq(focus_team)) &
                         ~matches.match_id.eq(match_id)].copy()
    if candidates.empty:
        st.info("No other match for this team is available in the selected StatsBomb season.")
    else:
        compare_label = st.selectbox("Comparison match", candidates.label.tolist())
        compare_match = candidates[candidates.label.eq(compare_label)].iloc[0]
        try:
            compare_events, _ = load_match(int(compare_match.match_id))
            compare_teams = [compare_match.home_team, compare_match.away_team]
            current_row = comparative_coaching_summary(all_events, teams).query("team == @focus_team").iloc[0]
            other_row = comparative_coaching_summary(compare_events, compare_teams).query("team == @focus_team").iloc[0]
            metrics_to_compare = ["possession estimate %", "shots", "shots on target", "xG", "xG per shot",
                                  "pass completion %", "progressive passes", "final-third entries",
                                  "penalty-area entries", "pressures", "recoveries"]
            comparison_rows = []
            for metric in metrics_to_compare:
                current_value = float(current_row.get(metric, 0))
                other_value = float(other_row.get(metric, 0))
                comparison_rows.append({"metric": metric, "current match": current_value,
                                        "comparison match": other_value, "difference": current_value - other_value})
            st.table(format_table(pd.DataFrame(comparison_rows), hide_index=True))
            st.caption(f"Comparison: {selected_label} against {compare_label}. Positive differences mean a higher value in the current match.")
        except Exception as exc:
            st.error(f"The comparison match could not be loaded: {exc}")

with tabs[8]:
    st.subheader("Two-team coaching summary")
    st.caption(f"Comparison for minutes {minute_range[0]}–{minute_range[1]}. Select a detailed analysis team in the sidebar for the other tabs.")
    comparison = comparative_coaching_summary(events, teams)
    st.table(format_table(comparison, hide_index=True))

    left, right = st.columns(2)
    for column, team, other in [(left, home_team, away_team), (right, away_team, home_team)]:
        with column:
            st.markdown(f"### {team}")
            for note in coaching_observations(events, team, other):
                st.info(note)
    st.subheader(f"Phase comparison · {focus_team}")
    phase_rows = []
    phase_ranges = [("First half", 0, 45), ("Second half", 46, end_minute)]
    if first_sub is not None:
        phase_ranges.extend([("Before first substitution", 0, first_sub),
                             ("After first substitution", first_sub, end_minute)])
    for phase, start, finish in phase_ranges:
        phase_events = all_events[pd.to_numeric(series_or_default(all_events, "minute", 0), errors="coerce").between(start, finish)]
        row = team_summary(phase_events, teams).loc[focus_team].to_dict()
        row.update(pass_metrics(phase_events, focus_team))
        phase_rows.append({"phase": phase, **row})
    phase_table = pd.DataFrame(phase_rows)[["phase", "shots", "xG", "pass completion %",
                                            "progressive passes", "final-third entries", "penalty-area entries"]]
    st.table(format_table(phase_table, hide_index=True))

with tabs[9]:
    st.subheader("Build coaching report")
    report_team = st.selectbox("Report team", teams, index=teams.index(focus_team), key="report_team")
    report_opponent = away_team if report_team == home_team else home_team
    generate_col1, generate_col2 = st.columns(2)
    analysis_length = generate_col1.selectbox("Analysis length", ["Brief", "Standard", "Detailed"], index=1)
    analysis_focus = generate_col2.selectbox("Coaching focus", ["Complete", "In possession", "Out of possession", "Transitions"])
    if st.button("Generate match analysis", type="primary", use_container_width=True):
        generated = generated_match_analysis(events, report_team, report_opponent, analysis_length, analysis_focus)
        st.session_state.generated_analysis_headings = list(generated)
        for heading, content in generated.items():
            st.session_state[f"generated_edit_{heading}"] = content
        st.session_state["strengths"] = generated.get("Strengths", st.session_state.get("strengths", ""))
        st.session_state["priorities"] = generated.get("Development priorities", st.session_state.get("priorities", ""))
        st.session_state["training"] = generated.get("Training recommendations", st.session_state.get("training", ""))
        st.session_state.pop("report_pdf", None)

    generated_sections = {}
    if st.session_state.get("generated_analysis_headings"):
        st.markdown("#### Generated analysis")
        for heading in st.session_state.generated_analysis_headings:
            st.markdown(f"**{heading}**")
            generated_sections[heading] = st.text_area(
                heading, key=f"generated_edit_{heading}", label_visibility="collapsed",
                height=100 if analysis_length != "Detailed" else 120,
            )

    st.markdown("#### Coach inputs")
    match_plan = st.text_area("Match plan and objectives", key="match_plan")
    strengths = st.text_area("Strengths", key="strengths")
    priorities = st.text_area("Development priorities", key="priorities")
    video = st.text_area("Video evidence and timestamps", key="video")
    training = st.text_area("Training response", key="training")
    include_analysis = st.checkbox("Include generated analysis", value=True,
                                   disabled=not bool(generated_sections))
    include_shots = st.checkbox("Include shot map", value=True)
    include_network = st.checkbox("Include pass network", value=True)
    if st.button("Prepare PDF report", type="primary"):
        pdf_summary = comparative_coaching_summary(events, teams)
        pdf_notes = {team: coaching_observations(events, team, away_team if team == home_team else home_team)
                     for team in teams}
        pdf_figures = []
        if include_shots:
            pdf_figures.append(("Shot map", shot_map(events[series_or_default(events, "type").eq("Shot")], teams)))
        if include_network:
            pdf_figures.append((f"{report_team} pass network",
                                pass_network(events, report_team, lineups, minimum_connection)))
        st.session_state.report_pdf = make_pdf_report(
            f"{home_team} {int(match.home_score)}–{int(match.away_score)} {away_team}", minute_range,
            pdf_summary, pdf_notes,
            {"match_plan": match_plan, "strengths": strengths, "priorities": priorities,
             "video": video, "training": training,
             "generated_sections": generated_sections if include_analysis else {}}, pdf_figures,
        )
    if "report_pdf" in st.session_state:
        st.download_button("Download coaching report PDF", st.session_state.report_pdf,
                           f"{match_id}_{report_team}_coaching_report.pdf", "application/pdf")

with tabs[10]:
    st.subheader("Event data")
    event_options = sorted(series_or_default(all_events, "type").dropna().astype(str).unique())
    selected_types = st.multiselect("Event types", event_options, default=event_options)
    displayed_events = all_events[series_or_default(all_events, "type").isin(selected_types)].copy()
    display_columns = [column for column in ["minute", "second", "period", "team", "player", "type",
                                                    "pass_recipient", "pass_outcome", "shot_outcome",
                                                    "shot_statsbomb_xg", "x", "y", "end_x", "end_y", "notes"]
                       if column in displayed_events.columns]
    st.write(f"Showing **{len(displayed_events)}** of **{len(all_events)}** event rows")
    st.table(format_table(displayed_events[display_columns], hide_index=True))
    st.download_button("Download normalised event data", displayed_events[display_columns].to_csv(index=False).encode("utf-8"),
                       f"{match_id}_normalised_events.csv", "text/csv")

with st.expander("Data notes and limitations"):
    st.markdown(
        """
        - StatsBomb open data covers selected competitions and matches rather than every team.
        - Uploaded files need one row per event for tactical charts. Aggregate reports can still be viewed and downloaded.
        - Coordinate-based charts require start locations. Passing progression also requires end locations.
        - Provider exports vary by subscription, software version and tagging setup; review the field and event-name mappings after upload.
        - Possession is estimated from event durations, with possession sequences used as a fallback.
        - Average positions use completed-pass origins; they are not tracking-data positions.
        - A recorded pressure is not the same as a complete measure of pressing intensity.
        - Event counts describe on-ball actions and cannot fully represent off-ball tactical roles.
        """
    )
