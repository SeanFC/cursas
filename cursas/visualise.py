#TODO: Add record times (world record and PR record) to relevant plots
#TODO: Sex and gender naming mixing 
#TODO: Add a demographics plot (the distributions of age and sex)
#TODO: Red blue colour scheme doesn't really go with the rest of the website. Need a different accent colour than red.
#TODO: Need some explanation text of under all the plots

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
        data = self.get_statistics()
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Scatter(x=data['Date'], y=data['Attendance'], name="Attendance"), secondary_y=False)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Races Run'], name="Races Run"), secondary_y=True)

        fig.update_layout(
                title="Attendance and Number of Events Through Time",
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
                hovermode="x unified" #TODO: The title of this doesn't include the day
                )
        return fig

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

        colour_scheme = px.colors.sequential.Bluered
       
        #Note: The function to create colours wont work for any given colour scheme because the below will only work for rgb(r,g,b) colour strings, not other colour specification types.
        colours = { 
                str(y):col
                for y, col in zip(
                    np.arange(int(data['Year'].min()), int(data['Year'].max())+1), 
                    px.colors.n_colors(
                        colour_scheme[0], 
                        colour_scheme[-1],
                        n_colors=int(data['Year'].max()) - int(data['Year'].min())+1, 
                        colortype='rgb')
                    ) 
                }

        fig = px.scatter(data, x='Day of Year', y='Time', color='Year', hover_name='Date', 
                color_discrete_map = colours,
                )

        month_lengths = np.array([monthrange(2011, month_idx)[1] for month_idx in range(1,13)])

        #TODO: Can't get these axis labels to be in the middle of the months
        fig.update_layout(
            title="Average Parkrun Race Results Through Time",
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

        time_edges = np.arange(0, 60*90, 60)/60
        men_hist, edges = np.histogram(results[results['Sex'] == 'M']['Time']/60, bins=time_edges)
        women_hist, edges = np.histogram(results[results['Sex'] == 'W']['Time']/60, bins=time_edges)
        
        out_df = pd.DataFrame({
            'Time': edges[:-1],
            'Men': men_hist,
            'Women': women_hist,
            })
        out_df = out_df[(out_df['Men'] > 0) | (out_df['Women'] > 0)] #TODO: This approach wont deal well with outliers

        return out_df

    def get_figure(self):
        #TODO: Unnamed: 0 column name
        data = self.get_statistics()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=x, y=y,
            text=y,
            textposition='auto',
        )])
        fig = px.bar(data, x='Time', y=['Men', 'Women'], hover_data=data) 

        step_size = 1 #TODO: This assumes step size
        hist_group_boundries = np.arange(data['Time'].min(), data['Time'].max()+step_size, step_size) 
        print(hist_group_boundries)
        print(data.dtypes)

        fig.update_layout(
            title="Distribution of Times for Adult Runners",
            xaxis_title="Time (minutes)",
            yaxis_title="Amount of Times Registered",
            legend_title="Sex",
            xaxis=dict(
                tickmode='array',
                tickvals=hist_group_boundries - step_size/2,
                ticktext=list(map(lambda x:str(int(x)), hist_group_boundries)),
                ),
            barmode='overlay'
        )
        fig.update_traces(opacity=0.75)
        #fig.show()
            
        return fig

        pass

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
            title="Distribution of Athletes That Completed a Number of Runs"
            )

        return fig


class AverageAttendancePlot(CachedStatFigure):
    def __init__(self, database):
        super().__init__(database, 'average_attendance.csv') 

    #TODO: Dropping nan means problem in dataset
    def create_statistics_table(self):
        # Pull from database
        events_df = self.database.get_all_run_events()
        attendance = self.database.get_all_results().groupby('Event ID').count()['Time']

        # Get a list of all the event runs with the data as how many days since the first event
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

        event_tracks = event_tracks.pivot_table(values='Attendance', index='Days since start', columns='Run ID')
        event_tracks = event_tracks.apply([
                lambda x: st.gmean(x[~np.isnan(x)]),
                np.nanstd,
                'count'
                ], axis=1).rename(columns={'<lambda>':'Mean Attendance', 'nanstd': 'Std Attendance'})
        event_tracks = event_tracks[event_tracks['count'] >= 2] #TODO: This needs to be mentioned in the output/function name/title
        event_tracks.rename(columns={'nangmean':'Mean Attendance', 'nanstd': 'Std Attendance', 'count':'Sample Size'}, inplace=True)
        event_tracks['Days since start'] = event_tracks.index #TODO: This creates an Unnamed: 0 Column

        return event_tracks
        
    def get_figure(self):
        data = self.get_statistics()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=data['Days since start'], 
            y=data['Mean Attendance'],
            name='Attendance'
            ),
            secondary_y=False
            )

        std_bound = np.array((data['Mean Attendance'] + data['Std Attendance']/2).tolist() + (data['Mean Attendance'] - data['Std Attendance']/2).tolist()[::-1])
        std_bound = np.max(np.vstack((std_bound, [0]*len(std_bound))), axis=0)

        fig.add_trace(go.Scatter(
            x=data['Days since start'].tolist() + data['Days since start'].tolist()[::-1],
            y=std_bound,
            fill='toself',
            line_color='blue',
            line_width=0,
            name='Attendance Spread',
            ), 
            secondary_y=False
            )

        fig.add_trace(go.Scatter(
            x=data['Days since start'],
            y=data['Sample Size'],
            name='Sample Size',
            line_color='red',
            ), 
            secondary_y=True
            )

        # X-axis should be by year
        fig.update_layout(
            title="Average Attendance at Parkrun",
            xaxis=dict(showgrid=True, title='Days Since Start'),
            yaxis=dict(showgrid=True, title='Average Attendance'),
            yaxis2=dict(title='Sample Size'),
            hovermode='x unified',
            showlegend=True
        ) 
        fig.update_yaxes(range=[10, None])

        return fig

#TODO: This needs to go somewhere
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
            title=f"The Performance Of {subject_name.title()} At Eastville Parkrun".title(),
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
        self.title = "Average and Fastest Times for Groups of Runners for the Last Year"

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

class EventAvgYear(CachedStatFigure):
    def __init__(self, database):
        super().__init__(database, 'event_avg_year.csv')

    def create_statistics_table(self):
        events = self.database.get_all_events()
        run_events = self.database.get_all_run_events()
        results = self.database.get_all_results()

        run_event_stats = results[['Time', 'Event ID']].groupby('Event ID').agg(**{
                'Attendance':pd.NamedAgg(column='Event ID', aggfunc='count'),
                'Time':pd.NamedAgg(column='Time', aggfunc=np.mean)
                })
        run_event_stats = run_event_stats.merge(run_events, how='inner', left_on='Event ID', right_index=True)

        cutoff_date = run_event_stats['Date'].max()
        cutoff_date = cutoff_date - pd.Timedelta("366 day")
        run_event_stats = run_event_stats[run_event_stats['Date'] > cutoff_date]
        
        out_df = run_event_stats[['Attendance', 'Time', 'Run ID']].groupby('Run ID').apply(np.mean)
        return out_df.merge(events, how='inner', left_on='Run ID', right_index=True).rename(columns={'Display Name':'Event Display Name'})

    def get_figure(self):
        data = self.get_statistics()
        data['Time']/=60

        fig = px.scatter(data, x='Attendance', y='Time', hover_name='Event Display Name', 
                hover_data=['Attendance', 'Time'],
                marginal_x='histogram', marginal_y='histogram'
                ) 

        fig.update_layout(
            title="Different Runs Over the Last 12 Months",
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
    # Explore Simulated Parkrun Data

    **[Parkrun](https://www.parkrun.com/)** is a series of free to enter 5 kilometre running races held weekly across the world. 
    This website shows interactive analysis on a **[simulated](#simulated)** version of the finish times for these races.

    This website has no affiliation with Parkrun and so has no access to the real data however, if you think you could help change this please **[contact me](https://www.sfcleator.com/contact)**.

    ## Parkrun as a Whole
    '''),
    dcc.Graph(figure=OverallAttendanceEventsPlot(database).get_figure()),
    dcc.Graph(figure=YearlyAverageTimePlot(database).get_figure()),
    dcc.Markdown('''
    ## Simulated Dataset

    Instead of using the real Parkrun data this project instead simulates completely fictional races, athletes and run times.
    The distributions of various aspects of the dataset (e.g. sex ratios, amount of races run etc.) are designed to mimic what I think would appear in the real dataset.
    However, the information of specific events or athletes isn't designed to match the real dataset and so any similarities are purely coincidental.
    Note this is most relevant in the way that the event names may be similar to the real event names however, any specific data about these events isn't designed to match the real thing. 

    ### Why?

    Although the real data is available on the Parkrun website it is asked that you don't scrape it.
    Further, the real data may contain personally identifiable data which generally requires extra care.
    Hence, I've opted to build up the data analysis infrastructure seen here with the hope of working on similar real data in the future.

    ### How?

    A rough description of how the data is simulated can be explained in four main steps:

    1. A set of events is generated - Each event has a typical name and date of first event. The distribution of the date of first events is set to start small and increase over time.
    2. For each event a set of 'run events', a race that took place on some Saturday, is generated. These start at the date of the first event and go on Saturday to the present day. Random Saturdays are skipped to simulated cancelled events.
    3. A set of athletes are generated with a sex, age group, home event, typical result time, first participated event and number of event participations. The distribution of athlete attributes is selected to be a basic approximation of what is expected in the real data.
    4. For each athlete a result time is generated for a simulated list of run events that the athlete took part in. The time is randomly generated based on the athlete's attributes (age, typical result time, etc.). The list of run events is a random selection of the run events at the athlete's home event that the athlete could feasibly take part in (considering when their first event was and how many runs they've done.

    This isn't the full process for how the dataset is created as additional steps are taken to ensure that unrealistic data doesn't creep in such as someone competing in two events as once. 
    Further, this is a very crude simulation of the real data and it is possible this simulation will improve in the future.
    Some of the bigger problems include:

    * It is likely that result times are much more nuanced than the model used here.
    * Athletes only run at their home event .
    * Several of the distributions used are either unrealistic (e.g. the age groups are poorly distributed).
    * Features of the data such as result times and attendance don't take into account extra factors such as course difficulty or weather.

    ### Can I see this analysis on the real data? 

    Without using the real data this isn't currently possible, however, with real Parkrun data this website could:

    * Make predictions about attendance and running times
    * Showcase the public's ability to run a 5K race
    * Help answer questions about common technical about Parkrun

    I'm very keen to discuss opportunities for performing these sorts of analyses on the real Parkrun data or similar datasets.
    If you work for/with Parkrun or work like to discuss how to apply this to your dataset please **[contact me](https://www.sfcleator.com/contact)**.
    ''', id='simulated'),
    ]

def get_average_event_tab(database):
    return [
            dcc.Markdown('''
            ## Statistics of Events

            An event describes the location, management team, etc. that runs races on most Saturdays.
            Use this page to compare all the different events against each other and understand what a typical event looks like statistically.

            ### Event Comparison
            '''),
            dcc.Graph(figure=EventAvgYear(database).get_figure()),
            #TODO: Add in-line sex ratio and age distribution scatter plot (y axis men/women ratio, x axis average age?)
            dcc.Markdown('### Typical Event'),
            dcc.Graph(figure=AverageAttendancePlot(database).get_figure()), 
            dcc.Markdown('''
            The geometric mean of the attendance at at event since the start of the event. 
            Geometric mean is used here instead of the standard mean is to account for outliers such as very popular events.
            
            The spread is shown as one standard deviation around the mean.
            The abrupt changes in spread are likely due to that fact that outlier events haven't run enough races rather than a change in likelihood of the mean being correct.
            '''),
            ]

def get_average_runner_tab(database):
    return [
            dcc.Markdown('''
            ## Statistics of Athletes 
            
            Use this page to explore the demographics and race results of all the different athletes.
            
            '''),
            dcc.Graph(figure=RunnerTimeDistributionPlot(database).get_figure()), 
            #dcc.Graph(figure=RunnerGroupsPlot(database).get_figure()),
            dcc.Graph(figure=RunnerGroupsYearPlot(database).get_figure()),
            dcc.Graph(figure=OverallRunAmountsPlot(database).get_figure()),
            #dcc.Graph(figure=plot_single_performance()), #TODO: Move to single runner lookup tab
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
                    dcc.Tab(label='Events', children=get_average_event_tab(database),
                        style=unselected_tab_style,
                        selected_style=selected_tab_style
                        ),
                    dcc.Tab(label='Athletes', children=get_average_runner_tab(database),
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

    #dev_figure_class.create_statistics()

    app.layout = html.Div(className="grid-container", children=[
        dcc.Graph(figure=dev_figure_class.get_figure())
        ])
    return app
