#TODO: Move out mpl
#TODO: Pre-calculate all of these statistics ahead of time

import random
import time
import pickle as pkl

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
                                                                                          
from cursas.config import *
from cursas.extract import generate_full_run_table
from cursas.extract import RunEvent, ResultEntry


def plot_all_row_entry_times():
    with open(full_table_file_name, 'rb') as f:  
        _, all_rows = pkl.load(f)

    data_table = np.array([
        [(row.time), 'M' in row.age_group, row.age_group] for row in all_rows if (row.athlete_id != -1)
        ])
    print(data_table)
    
    age_groups = np.unique(data_table[:, 2])
    age_groups = [ g for g in age_groups if 'M' in g]
    sections = [
            [ g for g in age_groups if 'J' in g],
            [ g for g in age_groups if 'S' in g],
            [ g for g in age_groups if 'V' in g],
            ]
    print(age_groups)
    print(sections)

    #times = data_table[:, 0].astype(int)
    #plt.hist(times/60, bins=400, histtype='step', label=group)

    for group in sections:
        times = data_table[np.any(data_table[:, 2] == group)][:, 0].astype(int)
        plt.hist(times/60, bins=400, histtype='step', label=group, density=True)
    plt.legend()
    plt.show()


    print(output)

def mpl_plot_yearly_average_time():
    #TODO: Some of the times of the last rows are being parse incorrectly
    with open(full_table_file_name, 'rb') as f:  
        all_events, all_rows = pkl.load(f)
    
    data_table = np.array([
        [int(row.time), int(row.event_id)] for row in all_rows if (row.athlete_id != -1)
        ])

    event_ids = np.unique(data_table[:, 1])
    event_date_lookup = dict( (x.event_id, x.date) for x in all_events)

    avg_times = []
    for e_id in event_ids:
        cur_times = np.array(data_table[data_table[:, 1] == e_id][:, 0], dtype=float)
        cur_times[cur_times <= 7*60] = np.nan

        avg_times.append([
            event_date_lookup[f'{e_id}'], 
            np.nanmean(cur_times),
            np.nanmin(cur_times)
            ])
    avg_times = np.array(avg_times)

    fig, axes = plt.subplots(2, 1, sharex=True)
    avg_ax = axes[0]
    min_ax = axes[1]

    for year in range(2017, 2021):
        vals_this_year = np.array([ x for x in avg_times if x[0].year == year])
        dates_this_year = [ int(x.strftime('%j')) for x in vals_this_year[:, 0]]
        avg_ax.scatter(dates_this_year, vals_this_year[:, 1]/60, label=year)
        min_ax.scatter(dates_this_year, vals_this_year[:, 2]/60, label=year)

    from calendar import monthrange, month_name 
    month_lengths = [0]
    for month_idx in range(1,12):
        month_lengths.append(monthrange(2011, month_idx)[1])
    avg_ax.set_xticks(np.cumsum(month_lengths))
    avg_ax.set_xticklabels(month_name[1:], rotation=45)
    avg_ax.legend(title='Year')
    avg_ax.grid()
    avg_ax.set_xlim([0, 366])
    fig.suptitle(all_events[0].run_name.title())
    avg_ax.set_ylabel('Average Parkrun Time (minutes)')

    from calendar import monthrange, month_name 
    month_lengths = [0]
    for month_idx in range(1,12):
        month_lengths.append(monthrange(2011, month_idx)[1])
    min_ax.set_xticks(np.cumsum(month_lengths))
    min_ax.set_xticklabels(month_name[1:], rotation=45)
    #min_ax.legend()
    min_ax.grid()
    min_ax.set_xlim([0, 366])
    fig.suptitle(all_events[0].run_name.title())
    min_ax.set_xlabel('Time of Year')
    min_ax.set_ylabel('Minimum Parkrun Time (minutes)')

    #plt.plot(avg_times[:, 0], avg_times[:, 2]/60)

    #times = data_table[:, 0].astype(int)
    #plt.hist(times/60, bins=400, histtype='step', label=group)

    plt.show()


def plot_yearly_average_time():
    #TODO: Some of the times of the last rows are being parse incorrectly
    with open(full_table_file_name, 'rb') as f:  
        all_events, all_rows = pkl.load(f)
    
    data_table = np.array([
        [int(row.time), int(row.event_id)] for row in all_rows if (row.athlete_id != -1)
        ])

    event_ids = np.unique(data_table[:, 1])
    event_date_lookup = dict( (x.event_id, x.date) for x in all_events)

    import datetime as dt

    #TODO: Not a great way to make this dataframe
    avg_times = []
    for e_id in event_ids:
        cur_times = np.array(data_table[data_table[:, 1] == e_id][:, 0], dtype=float)
        cur_times[cur_times <= 7*60] = np.nan
        avg_times.append([
            str(event_date_lookup[f'{e_id}'].year), # Cast to string here makes the plotting of the scatter separate into different groups
            (event_date_lookup[f'{e_id}'] - dt.date(event_date_lookup[f'{e_id}'].year, 1, 1)).days, 
            event_date_lookup[f'{e_id}'].strftime('%d %B %Y'), 
            np.nanmean(cur_times)/60,
            np.nanmin(cur_times)/60
            ])
    avg_times = np.array(avg_times)
    df = pd.DataFrame(avg_times, columns=['Event Year', 'Day of Year', 'Event Date', 'Mean Time', 'Min Time'])

    fig = px.scatter(df, x='Day of Year', y='Mean Time', color='Event Year', hover_name='Event Date') 

    from calendar import monthrange, month_name 
    month_lengths = np.array([ monthrange(2011, month_idx)[1] for month_idx in range(1,13)])

    #TODO: Can't get these axis labels to be in the middle of the months
    fig.update_layout(
        title="Average Eastville Parkrun times",
        xaxis = dict(
            tickmode='array',
            tickvals=np.array([0, *np.cumsum(month_lengths[:-1])]),# + month_lengths/2,
            ticktext=month_name[1:],
            showgrid=True,
            gridcolor='white',
        ),
        yaxis=dict(showgrid=False),
    ) 

    return fig
    #from calendar import monthrange, month_name 
    #month_lengths = [0]
    #for month_idx in range(1,12):
    #    month_lengths.append(monthrange(2011, month_idx)[1])
    #avg_ax.set_xticks(np.cumsum(month_lengths))
    #avg_ax.set_xticklabels(month_name[1:], rotation=45)
    #avg_ax.legend(title='Year')
    #avg_ax.grid()
    #avg_ax.set_xlim([0, 366])
    #fig.suptitle(all_events[0].run_name.title())
    #avg_ax.set_ylabel('Average Parkrun Time (minutes)')

    #from calendar import monthrange, month_name 
    #month_lengths = [0]
    #for month_idx in range(1,12):
    #    month_lengths.append(monthrange(2011, month_idx)[1])
    #min_ax.set_xticks(np.cumsum(month_lengths))
    #min_ax.set_xticklabels(month_name[1:], rotation=45)
    ##min_ax.legend()
    #min_ax.grid()
    #min_ax.set_xlim([0, 366])
    #fig.suptitle(all_events[0].run_name.title())
    #min_ax.set_xlabel('Time of Year')
    #min_ax.set_ylabel('Minimum Parkrun Time (minutes)')

    #plt.plot(avg_times[:, 0], avg_times[:, 2]/60)

    #times = data_table[:, 0].astype(int)
    #plt.hist(times/60, bins=400, histtype='step', label=group)

    plt.show()

def mpl_time_hiso_plot(times, data_table):
    mean, median, mode = (np.mean(times), np.median(times), st.mode(list(map(lambda x:int(x), times)))[0][0])
    
    fig, ax = plt.subplots(1,1)
    ax.hist(times, bins=np.arange(int(np.min(times)), int(np.max(times))+1))
    y_lims = ax.get_ylim()

    from matplotlib.offsetbox import AnchoredText
    anchored_text = AnchoredText(
            'Mean: '+ str(round(mean, 2) ) + 'm\n' + 
            'Median: '+ str(round(median, 2)) + 'm\n' +
            'Mode: '+ str(mode) + '-' + str(mode+1) + 'm'
            , 
            loc='upper right', frameon=False)
    ax.add_artist(anchored_text)

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Amount of times registered')
    ax.set_title('Eastville Parkrun times for women between 20 and 24 years old')

    ax.set_ylim(y_lims)
    plt.show()

def plot_female_21_24():
    with open(full_table_file_name, 'rb') as f:  
        all_events, all_rows = pkl.load(f)

    df = pd.DataFrame([], columns=['Time', 'Sex'])

    for code, sex in zip(['W', 'M'], ['Female', 'Male']):
        times = np.array([
            int(row.time) for row in all_rows if (row.athlete_id != -1) and (f'S{code}25-29' in row.age_group)
            ])
        times = times/60
        df = df.append(pd.DataFrame({'Time':times, 'Sex':[sex]*len(times)}), ignore_index=True)

    df = df[df['Time'] >3] #TODO: Clean up at data input stage
    
    fig = px.histogram(df, x='Time', color='Sex') #TODO: Need minute bins

    fig.update_layout(
        title="Eastville Parkrun times for runners between 25 and 29 years old",
        xaxis_title="Time (minutes)",
        yaxis_title="Amount of times registered",
        legend_title="Sex",
    )

    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.75)
    #fig.show()
        
    return fig

def plot_overall_run_amounts():
    with open(full_table_file_name, 'rb') as f:  
        all_events, all_rows = pkl.load(f)

    athlete_ids = np.array([
            int(row.athlete_id) for row in all_rows if row.athlete_id != -1
            ])
    counts = np.array([
        np.count_nonzero(athlete_ids == cur_id)
        for cur_id in np.unique(athlete_ids)
        ])
    binned_counts = np.histogram(counts, bins=np.arange(1, np.amax(counts)+1))
        
    df = pd.DataFrame({
        'Completed Runs': binned_counts[1][:-1],
        'Number of Runners': binned_counts[0],
        })
    fig = px.bar(df, x='Completed Runs', y='Number of Runners', log_y=True)

    for shirt_val in [50, 100, 250, 500]:
        if df['Completed Runs'].max()*1.1 > shirt_val:
            fig.add_vline(x=shirt_val)

    fig.update_layout(
        title="Eastville Parkrun distribution of runners that completed a number of runs"
        )

    return fig

def plot_attendance():
    with open(full_table_file_name, 'rb') as f:  
        all_events, all_rows = pkl.load(f)

    #TODO: Not a very clean way to do this
    event_ids = np.array([ int(row.event_id) for row in all_rows ])
    df = pd.DataFrame([{
        'Event ID':int(cur_id),
        'Attendance':np.count_nonzero(event_ids == cur_id)
        }
        for cur_id in np.unique(event_ids)
        ])

    import datetime as dt
    date_df = pd.DataFrame([{
        'Event ID':int(cur_event.event_id),
        'Time':(cur_event.date - dt.date(cur_event.date.year, 1, 1)).days,
        'Year':cur_event.date.year
        }
        for cur_event in all_events
        ])
    df = df.merge(date_df, on='Event ID')
    df.set_index('Event ID', inplace=True)

    fig = px.line(df, x='Time', y='Attendance', color='Year') #Note: Should probably be a bar chart really


    from calendar import monthrange, month_name 
    month_lengths = np.array([ monthrange(2011, month_idx)[1] for month_idx in range(1,13)])

    #TODO: Order these by event in the year so that each year can be compared together
    #TODO: Can't get these axis labels to be in the middle of the months
    fig.update_layout(
        title="Attendance at Eastville Parkrun",
        xaxis = dict(
            tickmode='array',
            tickvals=np.array([0, *np.cumsum(month_lengths[:-1])]),# + month_lengths/2,
            ticktext=month_name[1:],
            showgrid=True,
            gridcolor='white',
        ),
        yaxis=dict(showgrid=False),
    ) 
    return fig

def plot_single_performance():
    subject_name = 'Sean CLEATOR'

    with open(full_table_file_name, 'rb') as f:  
        all_events, all_rows = pkl.load(f)

    #TODO: Not a very clean way to do this
    event_ids = np.array([ int(row.event_id) for row in all_rows ])
    df = pd.DataFrame([{
        'Event ID':int(cur_row.event_id),
        'Run Time':cur_row.time/60,
        'Position':cur_row.position,
        }
        for cur_row in all_rows if cur_row.name == subject_name
        ])

    date_df = pd.DataFrame([{
        'Event ID':int(cur_event.event_id),
        'Time':cur_event.date
        }
        for cur_event in all_events
        ])

    df = df.merge(date_df, on='Event ID')
    df.set_index('Event ID', inplace=True)
    df.sort_values('Time', inplace=True)

    #fig = px.line(df, x='Time', y='Run Time') #Note: Should probably be a bar chart really

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    #x_axis_vals = df['Time']
    x_axis_vals = np.arange(df.shape[0], dtype=np.int) + 1
    fig.add_trace(go.Scatter(x=x_axis_vals, y=df['Run Time'], name="Run Time"), secondary_y=False)
    fig.add_trace(go.Scatter(x=x_axis_vals, y=df['Position'], name="Position"), secondary_y=True)

    fig.update_layout(
            title=f"The performance of {subject_name.title()} at Eastville Parkrun",
            xaxis=dict(
                title='Run Number',
                #title='Time',
                tickmode='array',
                tickvals=x_axis_vals,
                ),
            yaxis=dict(title='Run Time (m)'),
            yaxis2=dict(title='Position'),
            )

    return fig

def get_navbar():
    return html.Ul(
            className="topnav",
            children=[
                html.Li(html.A('Home', href='/')),
                html.Li(html.A('Cursas', href='', className="active")),
                html.Li(html.A('Repositories', href='/cgit')),
                html.Li(html.A('Contact', href='/contact.html'), style={'float':'right'}),
                ]
            )
def build_full_app(app):
    #TODO: Tabs don't change colour on hover
    default_tab_style = {
            'display': 'block', 
            'color': 'white',
            'text-align': 'center',
            'padding': '7px 8px', 
            'text-decoration': 'none',
            'border-width': '0px',
            'border-color': '#3275c8'
            }
    unselected_tab_style = {**default_tab_style, 'background-color': '#333'}
    selected_tab_style = {**default_tab_style, 'background-color': '#3275c8'}

    app.layout = html.Div(className="grid-container", children=[
        html.Div(className="main", children=[
            html.Div(className="header", children=[
                get_navbar(),
                dcc.Tabs([
                    dcc.Tab(label='Overview', children=[
                        dcc.Graph(figure=plot_female_21_24()),
                        ],
                        style=unselected_tab_style,
                        selected_style=selected_tab_style
                        ),
                    dcc.Tab(label='Location', children=[
                        dcc.Graph(figure=plot_yearly_average_time()),
                        dcc.Graph(figure=plot_attendance()),
                        dcc.Graph(figure=plot_overall_run_amounts()),
                        ],
                        style=unselected_tab_style,
                        selected_style=selected_tab_style
                        ),
                    dcc.Tab(label='Runner', children=[
                        dcc.Graph(figure=plot_single_performance()),
                        ],
                        style=unselected_tab_style,
                        selected_style=selected_tab_style
                        ),
                    ], 
                    style={
                        'background-color':'#333'
                        }
                    )
    ])])])

    return app

if __name__ == "__main__":
    plot_yearly_average_time()
