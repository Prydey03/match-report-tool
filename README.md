Match Report Tool

Match Report Tool is a Streamlit application for analysing football match-event data. It can load matches from StatsBomb open data or analyse uploaded CSV, TSV, Excel, JSON and Sportscode-style XML files.

The application provides match statistics, pitch graphics, player comparisons, opponent analysis, coaching observations and downloadable PDF reports. It also includes a custom CSV template for teams without an existing event-data provider.

THIS PROJECT CONTAINS A LIVE APPLICATION:

[Match Report Tool](https://match-report-tool-alohihqbw6v5ewwmta3zoa.streamlit.app/)

Features

StatsBomb open-data match selection

CSV, TSV, Excel, JSON and XML uploads

Automatic field and event-name recognition

Manual mapping for unfamiliar provider exports

Team-colour controls

Match-period and match-state filters

Match overview and line-ups

Shot maps and xG timeline

Progressive passes and territorial analysis

Final-third and penalty-area entries

Average positions and passing networks

Pressures, recoveries, interceptions, blocks and duels

Player and opponent analysis

Match-to-match comparisons

Generated, editable match analysis

Coaching-report PDF export

PNG and CSV downloads

Downloadable event-entry template and field guide

Project files

Keep these files in the same folder as the Jupyter Notebook:

match-report-tool/
├── app.py
├── fictional_dummy_match_data.csv
├── requirements.txt
└── README.md

Only app.py is required to run the application. The fictional dummy CSV is included for testing.

Install the required packages in Jupyter

Open a Jupyter Notebook in the project folder and run:

%pip install streamlit pandas numpy matplotlib statsbombpy mplsoccer openpyxl xlrd jinja2

Restart the notebook kernel after installation if Jupyter asks you to do so.

Run the application from Jupyter

Run this in a notebook cell:

import sys

!{sys.executable} -m streamlit run app.py




Test with the fictional match

Start the Streamlit application.

Select Upload match data.

Upload fictional_dummy_match_data.csv.

Leave the coordinate system as 120×80.

Leave the time-field unit as Minutes.

Select Each team attacks left to right.

Confirm the detected teams and colours.

The entirely fictional match is Aurora Forge FC 2–1 Velvet Comet Athletic. It contains passes, pressures, shots, xG, goals, defensive actions, fouls, cards and substitutions.

Use StatsBomb open data

Select StatsBomb open data.

Choose a competition and season.

Choose a match.

Select Generate report.

Select the team and analysis period in the sidebar.

StatsBomb open data covers selected competitions and matches. It does not include every club or competition.

Upload match data

Supported formats are:

.csv

.tsv

.xlsx

.xls

.json

.xml for Sportscode-style event files

After upload, the application attempts to identify each field automatically. If the file is recognised, the mapping controls remain collapsed. Open them only if a field or event has been interpreted incorrectly.

Essential event fields

Field

Purpose

minute

Event time in the match

team

Team responsible for the event

event_type

Pass, shot, pressure, recovery or another action

Pitch graphics also require x and y. Passing-direction and progression analysis requires end_x and end_y.

Coordinate systems

The importer accepts:

StatsBomb-style 120×80

percentage-style 0–100

normalised 0–1

The application converts imported coordinates into a 120×80 internal pitch. If a provider records both teams in one fixed physical direction, select Single fixed pitch direction so the away-team coordinates are reversed for analysis.

Use the custom CSV template

Select Upload match data, then download:

Team event template

Template field guide

The event template contains three example rows. Delete those rows before entering a real match.

Use one row per event. Optional fields can be left blank. Team colours can be supplied as hex values such as #6A3DB8.

Generate a match analysis

Open Report Builder and select:

report team

analysis length

coaching focus

Select Generate match analysis. The application creates evidence-based sections using the selected match data. Every section can be edited before export.

Generated sections may include:

match summary

in-possession analysis

out-of-possession analysis

transitions

key periods

player observations

strengths

development priorities

training recommendations

Add the coach's match plan, video timestamps and further observations before preparing the PDF.

Export a PDF report

In Report Builder:

Review and edit the generated analysis.

Complete the coaching-input fields.

Choose whether to include the analysis, shot map and pass network.

Select Prepare PDF report.

Select Download coaching report PDF.

Data interpretation

Possession is estimated from event durations, with possession sequences used as a fallback.

Average positions use completed-pass origins and are not tracking-data positions.

Pressure events do not provide a complete measure of pressing intensity.

Uploaded files need one row per event for tactical analysis.

Aggregate exports can be displayed, but they cannot produce event or location analysis.

Missing coordinates reduce the available pitch analysis.

Event data cannot fully describe off-ball positioning or tactical responsibilities.

Troubleshooting

streamlit is not recognised

Launch Streamlit through the active Jupyter Python environment:

import sys

!{sys.executable} -m streamlit run app.py

A module is missing

Install the missing package in a notebook cell with %pip install, then restart the kernel.

The browser does not open

Copy the local URL shown underneath the Streamlit command and paste it into the browser.

An uploaded file produces no charts

Check that:

the required fields are mapped

event names are recognised

the file contains event-level rows

coordinate columns contain numbers

the correct coordinate scale is selected

the team names match the values in the uploaded file

The pitch orientation looks incorrect

Change Coordinate orientation in the upload settings.

The application keeps using an earlier report

Refresh the browser or clear the Streamlit cache from the application menu.

Streamlit Community Cloud deployment

To host the application, place the project files in a GitHub repository. Add a requirements.txt file containing:

streamlit
pandas
numpy
matplotlib
statsbombpy
mplsoccer
openpyxl
xlrd
jinja2

In Streamlit Community Cloud, create an app from the repository and set the entrypoint to:

app.py

Uploaded files and generated reports are session data and should not be treated as permanent storage.

Privacy

Match files may contain player names and performance information. Confirm that your club is authorised to upload and process the data, particularly when players are under 18. Avoid placing private match files or credentials in a public GitHub repository.
