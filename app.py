"""Streamlit coaching report for StatsBomb open data.

Run from Jupyter with:
    !streamlit run statsbomb_match_report.py
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from mplsoccer import Pitch
from statsbombpy import sb


st.set_page_config(page_title="StatsBomb Coaching Report", page_icon="⚽", layout="wide")

TEAM_COLOURS = {"primary": "#d71920", "secondary": "#111111"}
PITCH = Pitch(pitch_type="statsbomb", pitch_color="#2b7a3d", line_color="white")


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


def format_table(df: pd.DataFrame):
    return (
        df.rename(columns=lambda value: str(value).replace("_", " ").title())
        .style.set_properties(**{"text-align": "center"})
        .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])
    )


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def draw_and_show(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def match_end_minute(events: pd.DataFrame) -> int:
    if events.empty or "minute" not in events:
        return 90
    return max(90, int(pd.to_numeric(events["minute"], errors="coerce").max()))


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
        rows.append({
            "player": player,
            "touch events": len(pe),
            "passes": len(passes),
            "completed passes": int(series_or_default(passes, "pass_outcome").isna().sum()),
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
    colours = [TEAM_COLOURS["primary"], TEAM_COLOURS["secondary"]]
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
    return fig


def xg_timeline(shots: pd.DataFrame, teams: list[str]):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    colours = [TEAM_COLOURS["primary"], TEAM_COLOURS["secondary"]]
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
                  color=TEAM_COLOURS["primary"], edgecolors="white", ax=ax)
    for player, row in positions.iterrows():
        label = str(int(row.number)) if pd.notna(row.number) else str(player).split()[-1][:3]
        ax.annotate(label, (row.x, row.y), ha="center", va="center", color="white", weight="bold")
    ax.set_title(f"{team} · average completed-pass locations", color="white")
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
                  color=TEAM_COLOURS["primary"], edgecolors="white", ax=ax, zorder=2)
    for player, row in positions.iterrows():
        number = number_map.get(player, np.nan)
        label = str(int(number)) if pd.notna(number) else str(player).split()[-1][:3]
        ax.annotate(label, (row.x, row.y), ha="center", va="center", color="white", weight="bold", zorder=3)
    ax.set_title(f"{team} · completed-pass network (minimum {minimum})", color="white")
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
    passes = completed_passes(events, team)
    final_third = add_xy(passes)
    final_third_count = int((final_third.x >= 80).sum())
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
    if final_third_count:
        notes.append(f"There were {final_third_count} completed passes originating in the attacking third.")
    if pressures:
        notes.append(f"StatsBomb recorded {pressures} pressure events and {high_recovery_count} attacking-third recoveries.")
    return notes or ["The selected period is too limited for a strong rule-based observation."]


def comparative_coaching_summary(events: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    summary = team_summary(events, teams).reset_index()
    summary["xG per shot"] = (summary["xG"] / summary["shots"].replace(0, np.nan)).fillna(0).round(2)
    summary["pressures"] = summary.team.map(lambda team: count_type(events, "Pressure", team))
    summary["recoveries"] = summary.team.map(lambda team: count_type(events, "Ball Recovery", team))
    return summary


# -----------------------------------------------------------------------------
# Interface
# -----------------------------------------------------------------------------
st.title("⚽ StatsBomb Coaching Report")
st.caption("Interactive match analysis using StatsBomb open event data")

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
    matches["label"] = (matches.home_team + " vs " + matches.away_team + " · " +
                        matches.match_date.astype(str))
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

home_team, away_team = match.home_team, match.away_team
teams = [home_team, away_team]
end_minute = match_end_minute(all_events)

with st.sidebar:
    st.divider()
    st.header("Analysis filters")
    focus_team = st.radio("Team to analyse", teams)
    minute_range = st.slider("Minute range", 0, end_minute, (0, end_minute))
    minimum_connection = st.slider("Minimum passes in network", 2, 15, 5)
    st.info(f"Detailed tabs currently show **{focus_team}**")

events = all_events[
    pd.to_numeric(series_or_default(all_events, "minute", 0), errors="coerce").between(*minute_range)
].copy()
opponent = away_team if focus_team == home_team else home_team

st.header(f"{home_team} {int(match.home_score)}–{int(match.away_score)} {away_team}")
st.caption(f"{match.match_date} · analysis from minute {minute_range[0]} to {minute_range[1]}")

st.markdown(
    f"""
    <div style="background:#f2d338;color:#111;padding:12px 16px;border-left:7px solid #d71920;
                border-radius:5px;margin:8px 0 18px 0;font-size:1.05rem;">
        <strong>Detailed analysis team:</strong> {focus_team}<br>
        <span style="font-size:.9rem;">Opponent: {opponent} · Change this under Analysis filters in the sidebar.</span>
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

tabs = st.tabs(["Overview", "Attacking", "Possession", "Defending", "Players", "Timeline", "Coaching summary"])

with tabs[0]:
    st.subheader("Match overview")
    st.dataframe(summary, use_container_width=True)
    st.download_button("Download overview CSV", csv_bytes(summary.reset_index()), "match_overview.csv", "text/csv")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(home_team)
        st.dataframe(lineup_table(lineups, home_team, end_minute), hide_index=True, use_container_width=True)
    with c2:
        st.subheader(away_team)
        st.dataframe(lineup_table(lineups, away_team, end_minute), hide_index=True, use_container_width=True)

with tabs[1]:
    shots = events[series_or_default(events, "type").eq("Shot")].copy()
    st.subheader("Shot locations and chance quality")
    st.caption("Both teams are shown so you can compare where chances were created. Larger markers indicate higher-quality chances.")
    draw_and_show(shot_map(shots, teams))
    st.subheader("How expected goals developed")
    draw_and_show(xg_timeline(shots, teams))
    if not shots.empty:
        columns = [c for c in ["minute", "team", "player", "shot_outcome", "shot_statsbomb_xg", "shot_body_part", "shot_type"] if c in shots]
        st.dataframe(shots[columns], hide_index=True, use_container_width=True)
        st.download_button("Download shots CSV", csv_bytes(shots[columns]), "shots.csv", "text/csv")

with tabs[2]:
    st.subheader(f"Possession structure · {focus_team}")
    st.caption("These diagrams use completed-pass origins. They show on-ball structure, not the players’ complete off-ball positions.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Average pass locations")
        st.caption("Each marker shows a player's average location when completing a pass. Marker size reflects completed passes.")
        draw_and_show(average_positions(events, focus_team, lineups))
    with c2:
        st.markdown("#### Passing connections")
        st.caption(f"Lines join players who exchanged at least {minimum_connection} completed passes. Thicker lines mean more passes.")
        draw_and_show(pass_network(events, focus_team, lineups, minimum_connection))
    passes = events[series_or_default(events, "type").eq("Pass") & series_or_default(events, "team").eq(focus_team)]
    active_players = series_or_default(passes, "player").dropna().unique()
    st.markdown("#### Shirt-number key")
    st.dataframe(number_reference(lineups, focus_team, active_players), hide_index=True, use_container_width=True)
    st.download_button("Download team passes CSV", csv_bytes(passes), "team_passes.csv", "text/csv")

with tabs[3]:
    st.subheader(f"Defensive activity · {focus_team}")
    st.caption("The locations are where StatsBomb recorded each action. Use the colour-and-shape keys above the pitches.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Pressures")
        st.caption("Where a defender applied pressure to an opponent in possession.")
        draw_and_show(action_map(events, focus_team, ["Pressure"], "Pressure locations"))
    with c2:
        st.markdown("#### Defensive actions")
        st.caption("Recoveries, interceptions, blocks and duels are separated by colour and marker shape.")
        draw_and_show(action_map(events, focus_team,
                                 ["Ball Recovery", "Interception", "Block", "Duel"],
                                 "Defensive-action locations"))
    defensive_rows = []
    for team in teams:
        defensive_rows.append({"team": team, "pressures": count_type(events, "Pressure", team),
                               "recoveries": count_type(events, "Ball Recovery", team),
                               "interceptions": count_type(events, "Interception", team),
                               "duels": count_type(events, "Duel", team),
                               "fouls": count_type(events, "Foul Committed", team)})
    st.dataframe(pd.DataFrame(defensive_rows).set_index("team"), use_container_width=True)

with tabs[4]:
    stats = player_summary(events, focus_team)
    st.dataframe(stats, hide_index=True, use_container_width=True)
    st.download_button("Download player statistics CSV", csv_bytes(stats), "player_statistics.csv", "text/csv")
    if not stats.empty:
        selected_player = st.selectbox("Player action map", stats.player.tolist())
        player_events = events[series_or_default(events, "player").eq(selected_player)]
        draw_and_show(action_map(player_events, focus_team, player_events.type.dropna().unique(), f"{selected_player} events"))

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
        st.dataframe(frame[columns] if columns else pd.DataFrame(), hide_index=True, use_container_width=True)

with tabs[6]:
    st.subheader("Two-team coaching summary")
    st.caption(f"Comparison for minutes {minute_range[0]}–{minute_range[1]}. Select a detailed analysis team in the sidebar for the other tabs.")
    comparison = comparative_coaching_summary(events, teams)
    st.dataframe(comparison, hide_index=True, use_container_width=True)

    left, right = st.columns(2)
    for column, team, other in [(left, home_team, away_team), (right, away_team, home_team)]:
        with column:
            st.markdown(f"### {team}")
            for note in coaching_observations(events, team, other):
                st.info(note)

    st.markdown("### Turning this into a formal coaching report")
    st.markdown(
        """
        A useful exported report could follow five short sections:

        1. **Match plan and context** — intended formation, principles, opposition plan and the selected analysis period.
        2. **What the data shows** — score, xG, shot quality, passing structure, pressures and recoveries.
        3. **Video evidence** — attach timestamps to confirm or challenge each data-based observation.
        4. **Coaching conclusions** — retain no more than three strengths and three development priorities.
        5. **Training response** — connect each priority to a specific practice, coaching behaviour and success measure.

        The next development step would be a **Generate coaching report** form. You could enter the game plan,
        add or edit observations, record video timestamps and select training priorities. The app could then export
        those notes, the comparison table and chosen graphics into a consistent PDF report.
        """
    )
    st.caption("Automated observations use visible rules and should support—not replace—video review and coaching judgement.")

with st.expander("Data notes and limitations"):
    st.markdown(
        """
        - StatsBomb open data covers selected competitions and matches rather than every team.
        - Possession is estimated from event durations, with possession sequences used as a fallback.
        - Average positions use completed-pass origins; they are not tracking-data positions.
        - A recorded pressure is not the same as a complete measure of pressing intensity.
        - Event counts describe on-ball actions and cannot fully represent off-ball tactical roles.
        """
    )
