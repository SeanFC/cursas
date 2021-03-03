#TODO: Change name to Glaucus

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

def plot_yearly_average_time():
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


def plot_female_21_24(app):
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

    app.layout = html.Div([
        dcc.Graph(figure=fig)
    ])
    
    return app

if __name__ == "__main__":
    #grab_all_run_ids()

    #event_ids = parse_all_run_ids()
    #grab_all_runs_from_ids(event_ids)

    #generate_full_run_table('eastville')
    #plot_all_row_entry_times()
    #plot_yearly_average_time()
    app.run_server(debug=True, use_reloader=True)  

    #generate_full_run_table('eastville')
