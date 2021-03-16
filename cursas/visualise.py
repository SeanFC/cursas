#TODO: Pre-calculate all of these statistics ahead of time
#TODO: Add record times (world record and PR record) to relevant plots
#TODO: Sex and gender naming mixing 
#TODO: Use fig.update_layout(hovermode="x unified") where appropriate
#TODO: Add a demographics plot (the distributions of age and sex)

import random
import time
import pickle as pkl
import datetime as dt
from calendar import monthrange, month_name 
from abc import ABC, abstractmethod
import os.path as osp

import pandas as pd
import numpy as np
import scipy.stats as st

import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
                                                                                          
from cursas.extract import CursasDatabase

#TODO: Maybe this can be done with function decorators?
class CachedStatFigure(ABC):
    def __init__(self, database, given_fp):
        self.cache_file_path = database.data_dir + given_fp
        self.database = database
    
    def get_statistics(self):
        if not osp.isfile(self.cache_file_path):
           raise RuntimeError('Statistics have never been generated')
        
        return pd.read_csv(self.cache_file_path)

    def create_statistics(self):
        self.create_statistics_table().to_csv(self.cache_file_path)

    @abstractmethod
    def create_statistics_table(self):
        pass
    
    @abstractmethod
    def get_figure(self):
        pass

class YearlyAverageTimePlot(CachedStatFigure):
    def __init__(self, database):
        super().__init__(database, 'yearly_average_time_stats.csv') 

    def create_statistics_table(self):
        with open(full_table_file_name, 'rb') as f:  
            all_events, all_rows = pkl.load(f)
        
        data_table = np.array([
            [int(row.time), int(row.event_id)] for row in all_rows if (row.athlete_id != -1)
            ])

        event_ids = np.unique(data_table[:, 1])
        event_date_lookup = dict( (x.event_id, x.date) for x in all_events)

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

        return pd.DataFrame(avg_times, columns=['Event Year', 'Day of Year', 'Event Date', 'Mean Time', 'Min Time'])

    def get_figure(self):
        fig = px.scatter(df, x='Day of Year', y='Mean Time', color='Event Year', hover_name='Event Date') 

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



#TODO: Delete this function if integrated into a class
def plot_yearly_average_time():
    #TODO: Some of the times of the last rows are being parse incorrectly
    with open(full_table_file_name, 'rb') as f:  
        all_events, all_rows = pkl.load(f)
    
    data_table = np.array([
        [int(row.time), int(row.event_id)] for row in all_rows if (row.athlete_id != -1)
        ])

    event_ids = np.unique(data_table[:, 1])
    event_date_lookup = dict( (x.event_id, x.date) for x in all_events)


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

class RunnerTimeDistributionPlot(CachedStatFigure):
    def __init__(self, database):
        super().__init__(database, 'runner_time_dist.csv') 

    def create_statistics_table(self):
         
        results = self.database.get_all_results()[['Time', 'Athlete ID']]
        athletes = self.database.get_all_athletes()[['Sex']]
        results = results.merge(athletes, how='outer', left_on='Athlete ID', right_index=True).dropna()

        time_edges = np.arange(0, 60*90, 60)
        men_hist, edges = np.histogram(results[results['Sex'] == 'M']['Time'], bins=time_edges)
        women_hist, edges = np.histogram(results[results['Sex'] == 'W']['Time'], bins=time_edges)

        return pd.DataFrame({
            'Time':edges[:-1], #TODO: This should say the groups
            'Men': men_hist,
            'Women': women_hist,
            })


    def get_figure(self):
        fig = px.bar(self.get_statistics(), x='Time', y=['Men', 'Women']) 

        fig.update_layout(
            title="Distribution of times for adult runners",
            xaxis_title="Time (minutes)",
            yaxis_title="Amount of times registered",
            legend_title="Sex",
        )

        fig.update_layout(barmode='overlay')
        fig.update_traces(opacity=0.75)
        #fig.show()
            
        return fig

        pass




#TODO: Delete this function if it is all put in class
def plot_runner_time_distribution():
    with open(full_table_file_name, 'rb') as f:  
        all_events, all_rows = pkl.load(f)

    df = pd.DataFrame([], columns=['Time', 'Sex'])

    for code, sex in zip(['W', 'M'], ['Female', 'Male']):
        times = np.array([
            int(row.time) for row in all_rows if (row.athlete_id != -1) and (f'{code}' in row.age_group) and ('J' not in row.age_group) #TODO: Junior excluded here
            ])
        times = times/60
        df = df.append(pd.DataFrame({'Time':times, 'Sex':[sex]*len(times)}), ignore_index=True)

    df = df[df['Time'] >3] #TODO: Clean up at data input stage
    
    fig = px.histogram(df, x='Time', color='Sex') #TODO: Need minute bins

    fig.update_layout(
        title="Eastville Parkrun times for adult runners",
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

def plot_overall_attendance_and_events():
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

    date_df = pd.DataFrame([{
        'Event ID':int(cur_event.event_id),
        'Time':cur_event.date,#(cur_event.date - dt.date(cur_event.date.year, 1, 1)).days,
        }
        for cur_event in all_events
        ])
    df = df.merge(date_df, on='Event ID')
    df.set_index('Event ID', inplace=True)
    df['Number of Events'] = np.argsort(df['Time'])

    fig = px.line(df, x='Time', y=['Attendance', 'Number of Events']) #Note: Should probably be a bar chart really

    fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                    ])
                )
            )

    return fig

def plot_runner_groups_avg_fast_scatter():
    with open(full_table_file_name, 'rb') as f:  
        all_events, all_rows = pkl.load(f)

    all_row_info = np.empty((len(all_rows), 3), dtype=np.object)
    for row_idx, row in enumerate(all_rows): 
        #TODO: Junior excluded here
        #TODO: Don't know what C age group is? 
        if (row.athlete_id != -1) and (row.age_group != '') and ('J' not in row.age_group) and (row.age_group[2] != 'C'):
            #TODO: The separation between age group and gender should happen at the DB creation stage
            all_row_info[row_idx] = [
                    row.time/60, 
                    'Men' if row.age_group[1] == 'M' else 'Women', 
                    row.age_group[2:]
                    ]

    df = pd.DataFrame(all_row_info, columns=['Time', 'Sex', 'Age Group']) # Note that None rows are automatically ignored
    df = df[df['Time']>3].dropna() #TODO: Clean up at data input stage
    df['Time'] = pd.to_numeric(df['Time'])

    stats_df = df.groupby(['Sex', 'Age Group']).agg([
        'min',
        'mean', #TODO: Maybe median is better?
        'count'
        ]) 

    # Get rid of multi indices and columns
    stats_df = stats_df.reset_index()
    stats_df.columns = stats_df.columns.map(lambda x:x[0] if x[1] == '' else x[1])
    stats_df = stats_df.rename(columns={
        'min':'Fastest', 
        'mean':'Average', #TODO: Maybe median is better?
        'count':'Number of Records'
        })
    stats_df = stats_df[stats_df['Number of Records'] > 5] #TODO: Should include all information, create a better plot 
    
    fig = px.scatter(stats_df, x='Average', y='Fastest', size='Number of Records', color='Sex', hover_data=['Age Group']) #TODO: Need minute bins

    fig.update_layout(
        title="Average and Fastest Times for Groups of Runners",
        xaxis_title="Average Time (minutes)",
        yaxis_title="Fastest Time (minutes)",
    )

    return fig

#TODO: Very similar to plot above
def plot_runner_groups_avg_fast_scatter_12_month():
    with open(full_table_file_name, 'rb') as f:  
        all_events, all_rows = pkl.load(f)

    event_date_lookup = dict( (x.event_id, x.date) for x in all_events)
    cutoff_date = max(event_date_lookup.values())
    cutoff_date = dt.date(year=cutoff_date.year-1, month=cutoff_date.month, day=cutoff_date.day)

    all_row_info = np.empty((len(all_rows), 3), dtype=np.object)
    for row_idx, row in enumerate(all_rows): 
        #TODO: Junior excluded here
        #TODO: Don't know what C age group is? 
        if (row.athlete_id != -1) and (row.age_group != '') and ('J' not in row.age_group) and (row.age_group[2] != 'C'):
            if event_date_lookup[row.event_id] > cutoff_date:
                #TODO: The separation between age group and gender should happen at the DB creation stage
                all_row_info[row_idx] = [
                        row.time/60, 
                        'Men' if row.age_group[1] == 'M' else 'Women', 
                        row.age_group[2:],
                        ]


    df = pd.DataFrame(all_row_info, columns=['Time', 'Sex', 'Age Group']) # Note that None rows are automatically ignored
    df = df[df['Time']>3].dropna() #TODO: Clean up at data input stage
    df['Time'] = pd.to_numeric(df['Time'])

    stats_df = df.groupby(['Sex', 'Age Group']).agg([
        'min',
        'mean', #TODO: Maybe median is better?
        'count'
        ]) 

    # Get rid of multi indices and columns
    stats_df = stats_df.reset_index()
    stats_df.columns = stats_df.columns.map(lambda x:x[0] if x[1] == '' else x[1])
    stats_df = stats_df.rename(columns={
        'min':'Fastest', 
        'mean':'Average', #TODO: Maybe median is better?
        'count':'Number of Records'
        })
    stats_df = stats_df[stats_df['Number of Records'] > 5] #TODO: Should include all information, create a better plot 
    
    fig = px.scatter(stats_df, x='Average', y='Fastest', size='Number of Records', color='Sex', hover_data=['Age Group']) 

    fig.update_layout(
        title="Average and Fastest Times for Groups of Runners Over the Last 12 Months",
        xaxis_title="Average Time (minutes)",
        yaxis_title="Fastest Time (minutes)",
    )

    return fig

def plot_event_avg_time_and_runners_12_month():
    #TODO: scatter average time, amount of runners average over last 12 months for each event

    with open(full_table_file_name, 'rb') as f:  
        all_events, all_rows = pkl.load(f)

    event_date_lookup = {x.event_id:x.date for x in all_events}
    event_run_name_lookup = {x.event_id:x.run_name for x in all_events}
    cutoff_date = max(event_date_lookup.values())
    cutoff_date = dt.date(year=cutoff_date.year-1, month=cutoff_date.month, day=cutoff_date.day)

    all_row_info = np.empty((len(all_rows), 4), dtype='object') #TODO: This can be quite a big object
    for row_idx, row in enumerate(all_rows): 
        if (row.athlete_id != -1): #TODO: Should I be excluding these athlete ids
            all_row_info[row_idx] = [
                    row.time/60, 
                    row.event_id,
                    event_date_lookup[row.event_id],
                    event_run_name_lookup[row.event_id],
                    ]

    df = pd.DataFrame(all_row_info, columns=['Time', 'Event ID', 'Date', 'Run Name']).dropna() 
    df = df[df['Date'] > cutoff_date] # Only last 12 months
    df['Time'] = pd.to_numeric(df['Time'])

    attendance_counts = df['Event ID'].value_counts()
    avg_run_attendance = pd.DataFrame({'Attendance': attendance_counts, 'Run Name': [event_run_name_lookup[idx] for idx in attendance_counts.index.to_numpy()]})
    avg_run_attendance = avg_run_attendance.groupby(['Run Name']).mean()

    avg_run_time = df.groupby(['Run Name'])['Time'].mean()

    run_stats = avg_run_attendance.merge(avg_run_time, on='Run Name')
    run_stats['Run Name'] = run_stats.index.to_series().apply(lambda x:x.title()) #TODO: A bit sloppy with the titling

    fig = px.scatter(run_stats, x='Attendance', y='Time', hover_name='Run Name') 

    fig.update_layout(
        title="Different Runs over the Last 12 Months",
        xaxis_title="Average Attendance",
        yaxis_title="Average Time (minutes)",
    )
    return fig 

#TODO: Move the below stuff to it's own module 
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

def get_overview_tab():
    return [dcc.Markdown('''
    # Explore Parkun UK data

    Parkrun is a series of free to enter 5 kilometre running races run most Saturdays at 9AM across the world with many races taking places in the United Kingdom.
    Finish times are posted on the [Parkrun website](https://www.parkrun.org.uk/) and, after over TODO years of growing involvement, a large dataset has been generated which captures information about the public's running capability.
    This website presents a cross section of the data with the goal to:

    * Showcase the public's ability to run a 5K race
    * Help answer questions about Parkrun
    * Make predictions about attendance and running times

    ## Parkrun as a Whole

    Look at how Parkrun has grown from an initial group of runners to an international event.
    '''),
    dcc.Graph(figure=plot_overall_attendance_and_events()),
    ]

def get_average_event_tab():
    return [
            dcc.Markdown('## Statistics of an Average Event'),
            dcc.Graph(figure=plot_yearly_average_time()),
            dcc.Graph(figure=plot_attendance()),
            dcc.Graph(figure=plot_overall_run_amounts()),
            ]

def get_average_runner_tab():
    return [
            dcc.Markdown('## Statistics of a Typical Runner'),
            dcc.Graph(figure=plot_runner_time_distribution()), #TODO: Do for whole dataset and allow filtering?
            dcc.Graph(figure=plot_runner_groups_avg_fast_scatter()),
            dcc.Graph(figure=plot_runner_groups_avg_fast_scatter_12_month()),
            dcc.Graph(figure=plot_single_performance()), #TODO: Move to single runner lookup tab
            ]

def get_event_comparison_tab():
    return [
            dcc.Markdown('## Compare Different Events'),
            dcc.Graph(figure=plot_event_avg_time_and_runners_12_month()),
            ]

def build_full_app(app):
    #TODO: Tabs don't change colour on hover
    #TODO: Rather than setting these options for every tab it would probably be better to set some of them for the whole tabs structure
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
                    dcc.Tab(label='Overview', children=get_overview_tab(),
                        style=unselected_tab_style,
                        selected_style=selected_tab_style
                        ),
                    dcc.Tab(label='Average Event', children=get_average_event_tab(),
                        style=unselected_tab_style,
                        selected_style=selected_tab_style
                        ),
                    dcc.Tab(label='Average Runner', children=get_average_runner_tab(),
                        style=unselected_tab_style,
                        selected_style=selected_tab_style
                        ),
                    dcc.Tab(label='Event Comparison', children=get_event_comparison_tab(),
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

def build_dev_app(app, config):
    dev_figure_class = RunnerTimeDistributionPlot(
            CursasDatabase(config)
            )

    dev_figure_class.create_statistics()

    app.layout = html.Div(className="grid-container", children=[
        dcc.Graph(figure=dev_figure_class.get_figure())
        ])
    return app
