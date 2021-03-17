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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
                                                                                          
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
        events_df = self.database.get_all_run_events()
        results = self.database.get_all_results().merge(events_df, how='outer', left_on='Event ID', right_index=True).dropna()[['Date', 'Time']]
        results = results.groupby('Date').mean('Time')
        results.reset_index(inplace=True)

        results['Day of Year'] = results['Date'].apply(lambda x: (x.date()-dt.date(x.year,1,1)).days)
        results['Year'] = results['Date'].apply(lambda x: x.date().year)

        return results
       
    def get_figure(self):
        data = self.get_statistics()
        data['Time'] = data['Time']/60
        data['Year'] = data['Year'].astype(str)

        fig = px.scatter(data, x='Day of Year', y='Time', color='Year', hover_name='Date') #TODO: Year colours aren't great

        month_lengths = np.array([ monthrange(2011, month_idx)[1] for month_idx in range(1,13)])

        #TODO: Can't get these axis labels to be in the middle of the months
        fig.update_layout(
            title="Average Parkrun results",
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

class OverallRunAmountsPlot(CachedStatFigure):
    def __init__(self, database):
        super().__init__(database, 'overall_run_amounts.csv') 

    def create_statistics_table(self):
        results = self.database.get_all_results()#.groupby('Event ID').count()['Time']
        run_amounts = results['Athlete ID'].value_counts()
        bins = np.arange(1, run_amounts.max()+1)

        hist, _ = np.histogram(run_amounts, bins=bins)
        out_df = pd.DataFrame({
            'Completed Runs': bins[:-1],
            'Number of Runners': hist,
            })
        out_df.set_index('Completed Runs')
        return out_df

    def get_figure(self):
        data = self.get_statistics()
        fig = px.bar(data, x='Completed Runs', y='Number of Runners', log_y=True, range_y=[1, data['Number of Runners'].max()])

        for shirt_val in [50, 100, 250, 500]:
            if data['Completed Runs'].max()*1.1 > shirt_val:
                fig.add_vline(x=shirt_val)

        fig.update_layout(
            title="Distribution of Runners That Completed a Number of Runs"
            )

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

#TODO: Too much data in this plot
class AverageAttendancePlot(CachedStatFigure):
    def __init__(self, database):
        super().__init__(database, 'average_attendance.csv') 

    #TODO: Dropping nan means problem in dataset
    #TODO: Replace the Run ID with the run name
    def create_statistics_table(self):
        events_df = self.database.get_all_run_events()
        attendance = self.database.get_all_results().groupby('Event ID').count()['Time']

        event_tracks = events_df.merge(attendance, how='outer', left_index=True, right_index=True)
        event_tracks = event_tracks.merge(
                events_df.groupby('Run ID').min().rename(columns={'Date':'First Event'}),
                how='outer', left_on='Run ID', right_index=True
                )
        event_tracks['Days since start'] = event_tracks['Date'] - event_tracks['First Event']
        event_tracks['Days since start'] = event_tracks['Days since start'].astype(int)/60/60/24*1e-9
        del event_tracks['Date']
        del event_tracks['First Event']
        event_tracks.rename(columns={'Time':'Attendance'}, inplace=True)

        return event_tracks
        
    def get_figure(self):
        data = self.get_statistics()
        fig = px.line(data, x='Days since start', y='Attendance', color='Run ID', log_y=True) #Note: Should probably be a bar chart really

        #TODO: Order these by event in the year so that each year can be compared together
        #TODO: Can't get these axis labels to be in the middle of the months
        fig.update_layout(
            title="Attendance at Parkruns",
            yaxis=dict(showgrid=False),
        ) 
        fig.update_yaxes(range=[10, None])
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

class OverallAttendanceEventsPlot(CachedStatFigure):
    def __init__(self, database):
        super().__init__(database, 'attendance_and_events_overall.csv') 

    def create_statistics_table(self):
        #TODO:Dropping na here means we're dropping good debugging information. Lots of these events should have more runners.
        events_df = self.database.get_all_run_events()
        results = self.database.get_all_results().merge(events_df, how='outer', left_on='Event ID', right_index=True).dropna() 

        out_df = results.groupby(['Date']).count().merge(events_df.groupby(['Date']).count(), how='outer', left_on='Date', right_on='Date')
        out_df = out_df[['Athlete ID', 'Run ID_y']].dropna().sort_values('Date')
        out_df.rename(columns={'Athlete ID':'Attendance', 'Run ID_y':'Races Run'}, inplace=True)

        return out_df

    def get_figure(self):
        # Create figure with secondary y-axis
        data = self.get_statistics()
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        #x_axis_vals = df['Time']
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Attendance'], name="Attendance"), secondary_y=False)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Races Run'], name="Races Run"), secondary_y=True)

        fig.update_layout(
                title="Attendances and number of events at Parkrun",
                xaxis=dict(
                    title='Time',
                    rangeslider_visible=True,
                    rangeselector=dict(
                        buttons=list([
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(count=5, label="5y", step="year", stepmode="backward"),
                            dict(step="all")
                            ])
                        )
                    ),
                yaxis=dict(title='Attendance'),
                yaxis2=dict(title='Races Run'),
                )

        return fig

class RunnerGroupsPlot(CachedStatFigure):
    def __init__(self, database, data_file_name='runner_groups_avg_fast.csv'):
        super().__init__(database, data_file_name) 
        self.title = "Average and Fastest Times for Groups of Runners"
    
    def get_useable_results(self):
        results = self.database.get_all_results()[['Time', 'Athlete ID']]
        athletes = self.database.get_all_athletes()[['Sex', 'Age Group']]
        results = results.merge(athletes, how='outer', left_on='Athlete ID', right_index=True).dropna()
        del results['Athlete ID']
        results['Time'] = pd.to_numeric(results['Time'])

        return results

    def create_statistics_table(self):
        stats_df = self.get_useable_results().groupby(['Sex', 'Age Group']).agg([
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
        return stats_df
                
    def get_figure(self):
        data = self.get_statistics()
        data['Average']/=60
        data['Fastest']/=60
        fig = px.scatter(data, x='Average', y='Fastest', size='Number of Records', color='Sex', hover_data=['Age Group']) 

        fig.update_layout(
            title=self.title,
            xaxis_title="Average Time (minutes)",
            yaxis_title="Fastest Time (minutes)",
        )

        return fig

class RunnerGroupsYearPlot(RunnerGroupsPlot):
    def __init__(self, database):
        super().__init__(database, data_file_name='runner_groups_avg_fast.csv')
        self.title = "Average and Fastest Times for Groups of Runners for the last Year"

    def get_useable_results(self):
        results = self.database.get_all_results()[['Time', 'Athlete ID', 'Event ID']]
        athletes = self.database.get_all_athletes()[['Sex', 'Age Group']]
        events = self.database.get_all_run_events()

        results = results.merge(athletes, how='outer', left_on='Athlete ID', right_index=True).dropna()
        del results['Athlete ID']
        results = results.merge(events, how='outer', left_on='Event ID', right_index=True).dropna()

        cutoff_date = results['Date'].max()
        cutoff_date = cutoff_date - pd.Timedelta("366 day")
        results = results[results['Date'] > cutoff_date]

        results['Time'] = pd.to_numeric(results['Time']) #TODO: Is this line needed (also repeat in child function)

        return results[['Sex', 'Age Group', 'Time']]
    
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

def get_overview_tab(database):
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
    dcc.Graph(figure=OverallAttendanceEventsPlot(database).get_figure()),
    ]

def get_average_event_tab(database):
    return [
            dcc.Markdown('## Statistics of an Average Event'),
            dcc.Graph(figure=YearlyAverageTimePlot(database).get_figure()),
            dcc.Graph(figure=AverageAttendancePlot(database).get_figure()),
            dcc.Graph(figure=OverallRunAmountsPlot(database).get_figure()),
            ]

def get_average_runner_tab(database):
    return [
            dcc.Markdown('## Statistics of a Typical Runner'),
            dcc.Graph(figure=RunnerTimeDistributionPlot(database).get_figure()), 
            dcc.Graph(figure=RunnerGroupsPlot(database).get_figure()),
            dcc.Graph(figure=RunnerGroupsYearPlot(database).get_figure()),
            #dcc.Graph(figure=plot_single_performance()), #TODO: Move to single runner lookup tab
            ]

def get_event_comparison_tab(database):
    return [
            dcc.Markdown('## Compare Different Events'),
            dcc.Markdown('TODO: make plot_event_avg_time_and_runners_12_month'),
            #dcc.Graph(figure=plot_event_avg_time_and_runners_12_month()),
            ]

def build_full_app(app, config):
    database = CursasDatabase(config)

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
                    dcc.Tab(label='Overview', children=get_overview_tab(database),
                        style=unselected_tab_style,
                        selected_style=selected_tab_style
                        ),
                    dcc.Tab(label='Average Event', children=get_average_event_tab(database),
                        style=unselected_tab_style,
                        selected_style=selected_tab_style
                        ),
                    dcc.Tab(label='Average Runner', children=get_average_runner_tab(database),
                        style=unselected_tab_style,
                        selected_style=selected_tab_style
                        ),
                    dcc.Tab(label='Event Comparison', children=get_event_comparison_tab(database),
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
    dev_figure_class = RunnerGroupsYearPlot(
            CursasDatabase(config)
            )

    dev_figure_class.create_statistics()

    app.layout = html.Div(className="grid-container", children=[
        dcc.Graph(figure=dev_figure_class.get_figure())
        ])
    return app
