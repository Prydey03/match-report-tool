import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsbombpy import sb
from mplsoccer import Pitch

st.set_page_config(page_title="Match Report Tool", layout="wide")

st.title("Match Report Tool")
st.write("Select a competition, season, and match to generate a full coaching report.")

def style_table(df):
    df = df.rename(columns=lambda x: str(x).title())
    return df.style.set_properties(**{'text-align': 'center'}).set_table_styles(
        [{'selector': 'th', 'props': [('text-align', 'center')]}]
    )

def style_lineup_table(df):
    df = df.rename(columns=lambda x: str(x).title())
    col_widths = {
        'Squad Number': '80px',
        'Player': '220px',
        'Position': '140px',
        'Status': '130px',
        'Minutes Played': '110px'
    }
    styles = [{'selector': 'th', 'props': [('text-align', 'center')]}]
    for col, width in col_widths.items():
        if col in df.columns:
            idx = df.columns.get_loc(col) + 1
            styles.append({'selector': f'th.col{idx-1}, td.col{idx-1}',
                            'props': [('width', width), ('max-width', width)]})
    return df.style.set_properties(**{'text-align': 'center'}).set_table_styles(styles)

@st.cache_data
def get_competitions():
    return sb.competitions()

comps = get_competitions()

competition_names = sorted(comps['competition_name'].unique())
selected_competition = st.selectbox("Competition:", competition_names)

available_seasons = comps[comps['competition_name'] == selected_competition]['season_name'].unique()
selected_season = st.selectbox("Season:", sorted(available_seasons, reverse=True))

comp_row = comps[(comps['competition_name'] == selected_competition) &
                  (comps['season_name'] == selected_season)].iloc[0]
competition_id = comp_row['competition_id']
season_id = comp_row['season_id']

@st.cache_data
def get_matches(competition_id, season_id):
    return sb.matches(competition_id=competition_id, season_id=season_id)

matches = get_matches(competition_id, season_id)
matches['match_label'] = matches['home_team'] + ' vs ' + matches['away_team'] + ' (' + matches['match_date'].astype(str) + ')'

selected_match_label = st.selectbox("Match:", matches['match_label'])
match_id = matches[matches['match_label'] == selected_match_label]['match_id'].values[0]

if st.button("Generate Report"):
    with st.spinner("Pulling match data..."):
        events = sb.events(match_id)
        match_info = matches[matches['match_id'] == match_id]
        lineups = sb.lineups(match_id)

    home_team = match_info['home_team'].values[0]
    away_team = match_info['away_team'].values[0]
    home_score = match_info['home_score'].values[0]
    away_score = match_info['away_score'].values[0]

    st.header(f"{home_team} {home_score} - {away_score} {away_team}")
    st.divider()

    pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='white')

    # --- Match Overview ---
    st.subheader("Match Overview")

    possession_counts = events.groupby('possession_team')['possession'].nunique()
    possession_pct = (possession_counts / possession_counts.sum() * 100).round(1)

    shots = events[events['type'] == 'Shot'].copy()
    shots_by_team = shots.groupby('team').size()
    goals_by_team = shots[shots['shot_outcome'] == 'Goal'].groupby('team').size()
    xg_by_team = shots.groupby('team')['shot_statsbomb_xg'].sum().round(2)

    overview_table = pd.DataFrame({
        'Possession %': possession_pct,
        'Shots': shots_by_team,
        'Goals': goals_by_team,
        'xG': xg_by_team
    }).fillna(0)

    overview_table['Goals'] = overview_table['Goals'].astype(int)
    overview_table['Shots'] = overview_table['Shots'].astype(int)
    overview_table['Possession %'] = overview_table['Possession %'].map('{:.1f}'.format)
    overview_table['xG'] = overview_table['xG'].map('{:.2f}'.format)

    st.table(style_table(overview_table))

    st.divider()

    # --- Starting XI / Substitutes / Minutes Played ---
    st.subheader("Lineups & Minutes Played")

    def to_minutes(timestr):
        if timestr is None or pd.isna(timestr):
            return None
        parts = str(timestr).split(':')
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 60 + int(minutes)
        elif len(parts) == 2:
            minutes, seconds = parts
            return int(minutes)
        return None

    def build_lineup_table(team_name):
        team_lineup = lineups[team_name]
        rows = []
        for _, player in team_lineup.iterrows():
            positions = player['positions']

            if not positions:
                status = 'Unused Substitute'
                minutes_played = 0
                position_name = '-'
            else:
                starter = any(p.get('start_reason') == 'Starting XI' for p in positions)
                status = 'Starter' if starter else 'Substitute'
                position_name = positions[0].get('position', '-')

                first_from = to_minutes(positions[0].get('from'))
                last_to_raw = positions[-1].get('to')
                last_to = to_minutes(last_to_raw) if last_to_raw else 90

                minutes_played = 0
                if first_from is not None and last_to is not None:
                    minutes_played = max(0, last_to - first_from)

            rows.append({
                'squad number': player['jersey_number'],
                'player': player['player_name'],
                'position': position_name,
                'status': status,
                'minutes played': minutes_played
            })

        df = pd.DataFrame(rows)
        status_order = {'Starter': 0, 'Substitute': 1, 'Unused Substitute': 2}
        df['sort_key'] = df['status'].map(status_order)
        df = df.sort_values('sort_key').drop(columns='sort_key').reset_index(drop=True)
        return df

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**{home_team}**")
        st.table(style_lineup_table(build_lineup_table(home_team)))
    with col2:
        st.write(f"**{away_team}**")
        st.table(style_lineup_table(build_lineup_table(away_team)))

    st.divider()

    # --- Shot Map ---
    st.subheader("Shot Map")

    shots['x'] = shots['location'].apply(lambda loc: loc[0])
    shots['y'] = shots['location'].apply(lambda loc: loc[1])

    fig, ax = pitch.draw(figsize=(12, 8))
    for team, color, marker in [(home_team, 'red', 'o'), (away_team, 'yellow', '^')]:
        team_shots = shots[shots['team'] == team]
        goals = team_shots[team_shots['shot_outcome'] == 'Goal']
        non_goals = team_shots[team_shots['shot_outcome'] != 'Goal']
        pitch.scatter(non_goals['x'], non_goals['y'], ax=ax, color=color, edgecolors='black',
                      s=80, alpha=0.5, marker=marker)
        pitch.scatter(goals['x'], goals['y'], ax=ax, color=color, edgecolors='black',
                      s=250, marker='*' if team == home_team else 'X')
    plt.title('Shot Map — Both Teams', fontsize=14)
    st.pyplot(fig)

    st.divider()

    # --- Average Positions / Formation ---
    st.subheader("Average Positions")

    def build_formation_map(team_name, dot_color, text_color):
        team_passes = events[(events['type'] == 'Pass') & (events['team'] == team_name)].copy()
        team_passes['x'] = team_passes['location'].apply(lambda loc: loc[0])
        team_passes['y'] = team_passes['location'].apply(lambda loc: loc[1])

        avg_locations = team_passes.groupby('player').agg({'x': 'mean', 'y': 'mean'})
        number_map = lineups[team_name].set_index('player_name')['jersey_number']
        avg_locations['jersey_number'] = avg_locations.index.map(number_map)

        fig, ax = pitch.draw(figsize=(10, 7))
        pitch.scatter(avg_locations['x'], avg_locations['y'], s=500, color=dot_color, edgecolors='black', ax=ax)
        for player, row in avg_locations.iterrows():
            ax.annotate(str(int(row['jersey_number'])), xy=(row['x'], row['y']),
                        ha='center', va='center', fontsize=12, color=text_color, weight='bold')
        ax.set_title(f'{team_name} — Average Positions', fontsize=13)
        return fig

    col3, col4 = st.columns(2)
    with col3:
        st.pyplot(build_formation_map(home_team, 'red', 'white'))
    with col4:
        st.pyplot(build_formation_map(away_team, 'black', 'yellow'))

    st.divider()

    # --- Pass Networks ---
    st.subheader("Pass Networks")

    def build_pass_network(team_name, line_color, dot_color, text_color):
        completed_passes = events[(events['type'] == 'Pass') &
                                   (events['team'] == team_name) &
                                   (events['pass_outcome'].isna())].copy()
        completed_passes['x'] = completed_passes['location'].apply(lambda loc: loc[0])
        completed_passes['y'] = completed_passes['location'].apply(lambda loc: loc[1])

        avg_locations = completed_passes.groupby('player').agg({'x': 'mean', 'y': 'mean', 'id': 'count'})
        avg_locations.columns = ['x', 'y', 'pass_count']

        pass_pairs = completed_passes.groupby(['player', 'pass_recipient']).size().reset_index(name='pass_count')

        number_map = lineups[team_name].set_index('player_name')['jersey_number']
        avg_locations['jersey_number'] = avg_locations.index.map(number_map)

        fig, ax = pitch.draw(figsize=(16, 10))
        for _, row in pass_pairs.iterrows():
            passer, recipient, count = row['player'], row['pass_recipient'], row['pass_count']
            if passer in avg_locations.index and recipient in avg_locations.index:
                x1, y1 = avg_locations.loc[passer, ['x', 'y']]
                x2, y2 = avg_locations.loc[recipient, ['x', 'y']]
                ax.plot([x1, x2], [y1, y2], color=line_color, linewidth=count/5, alpha=0.5, zorder=1)

        pitch.scatter(avg_locations['x'], avg_locations['y'], s=avg_locations['pass_count']*4,
                      color=dot_color, edgecolors='black', ax=ax, zorder=2)

        for player, row in avg_locations.iterrows():
            ax.annotate(str(int(row['jersey_number'])), xy=(row['x'], row['y']),
                        ha='center', va='center', fontsize=11, color=text_color, weight='bold', zorder=3)

        ax.set_title(f'{team_name} — Pass Network', fontsize=14)
        return fig

    st.pyplot(build_pass_network(home_team, 'blue', 'red', 'white'))
    st.pyplot(build_pass_network(away_team, 'yellow', 'black', 'yellow'))

    st.divider()

    # --- Defensive Summary ---
    st.subheader("Defensive Summary")

    tackles_by_team = events[events['type'] == 'Duel'].groupby('team').size()
    interceptions_by_team = events[events['type'] == 'Interception'].groupby('team').size()
    fouls_by_team = events[events['type'] == 'Foul Committed'].groupby('team').size()

    defense_table = pd.DataFrame({
        'Duels': tackles_by_team,
        'Interceptions': interceptions_by_team,
        'Fouls': fouls_by_team
    }).fillna(0).astype(int)
    st.table(style_table(defense_table))

    st.divider()

    # --- Event Timeline ---
    st.subheader("Key Events")

    goals_timeline = shots[shots['shot_outcome'] == 'Goal'][['minute', 'team', 'player']].reset_index(drop=True)
    st.write("**Goals**")
    st.table(style_table(goals_timeline))

    key_events = events[events['type'].isin(['Substitution', 'Bad Behaviour'])][['minute', 'team', 'player', 'type']].reset_index(drop=True)
    st.write("**Substitutions / Cards**")
    st.table(style_table(key_events))