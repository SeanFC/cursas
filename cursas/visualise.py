""" Create plotly plots for the Cursas database """

# Standard library imports
import datetime as dt
from calendar import monthrange, month_name
from abc import ABC, abstractmethod
import os.path as osp

# External imports
import pandas as pd
import numpy as np
import scipy.stats as st

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Note: Maybe this can be done with function decorators?
class CachedStatFigure(ABC):
    """
    A figure that works by first generating statistics and then generating a plot from the statistics.
    The statistics table that is generated is cached (saved to disk) and only recreated on request.
    """
    def __init__(self, database, given_fp):
        """
        Constructor

        :param database: The accessor for the Cursas database.
        :type database: CursasDatabase
        :param given_fp: The file path to save the cached statistics to.
        :type given_fp: string
        """
        self.cache_file_path = database.data_dir + given_fp
        self.database = database

    def get_statistics(self):
        """
        Get the statistics table for the figure.

        :return: The statistics table
        :rtype: pandas.DataFrame
        :raises RuntimeError: If no statistics have been made.
        """
        if not osp.isfile(self.cache_file_path):
            raise RuntimeError('Statistics have never been generated')

        return pd.read_csv(self.cache_file_path)

    def create_statistics(self):
        """
        Make and cache the statistics for the figure.
        """
        self.create_statistics_table().to_csv(self.cache_file_path)

    @abstractmethod
    def create_statistics_table(self):
        """
        Create the statistics table that will be cached.

        :return: The statistics to be cached.
        :rtype: pandas.DataFrame
        """

    @abstractmethod
    def get_figure(self):
        """
        Generate the figure from the cached statistics.

        :return: The resulting figure.
        :rtype: plotly.graph_objects.Figure
        """

class OverallAttendanceEventsPlot(CachedStatFigure):
    """ The attendance and amount of events held each week for the full database. """
    def __init__(self, database):
        """
        Constructor

        :param database: The accessor for the Cursas database
        :type database: CursasDatabase
        """
        super().__init__(database, 'attendance_and_events_overall.csv')

    def create_statistics_table(self):
        """
        Create the statistics table that can be used to make the figure.

        :return: The statistics to be cached.
        :rtype: pandas.DataFrame
        """

        #TODO:Dropping na here means we're dropping good debugging information. Lots of these events should have more runners.
        events_df = self.database.get_all_run_events()
        results = self.database.get_all_results().merge(events_df, how='outer', left_on='Event ID', right_index=True).dropna()

        out_df = results.groupby(['Date']).count().merge(
                events_df.groupby(['Date']).count(),
                how='outer',
                left_on='Date', right_on='Date'
                )
        out_df = out_df[['Athlete ID', 'Run ID_y']].dropna().sort_values('Date')
        out_df.rename(columns={'Athlete ID':'Attendance', 'Run ID_y':'Races Run'}, inplace=True)

        return out_df

    def get_figure(self):
        """
        Generate the figure from the cached statistics.

        :return: The resulting figure.
        :rtype: plotly.graph_objects.Figure
        """

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
    """ The average time every week, separated out into years in the database."""
    def __init__(self, database):
        """
        Constructor

        :param database: The accessor for the Cursas database
        :type database: CursasDatabase
        """
        super().__init__(database, 'yearly_average_time_stats.csv')

    def create_statistics_table(self):
        """
        Create the statistics table that can be used to make the figure.

        :return: The statistics to be cached.
        :rtype: pandas.DataFrame
        """
        events_df = self.database.get_all_run_events()
        results = self.database.get_all_results().merge(
                events_df, how='outer', left_on='Event ID', right_index=True
                ).dropna()[['Date', 'Time']]
        results = results.groupby('Date').mean('Time')
        results.reset_index(inplace=True)

        results['Day of Year'] = results['Date'].apply(lambda x: (x.date()-dt.date(x.year,1,1)).days)
        results['Year'] = results['Date'].apply(lambda x: x.date().year)

        return results

    def get_figure(self):
        """
        Generate the figure from the cached statistics.

        :return: The resulting figure.
        :rtype: plotly.graph_objects.Figure
        """

        data = self.get_statistics()
        data['Time'] = data['Time']/60
        data['Year'] = data['Year'].astype(str)

        colour_scheme = px.colors.sequential.Bluered

        # Note: The function to create colours wont work for any given colour scheme
        #       because the below will only work for rgb(r,g,b) colour strings, not other colour specification types.
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
    """ The distribution of result times for men and women. """
    def __init__(self, database):
        """
        Constructor

        :param database: The accessor for the Cursas database
        :type database: CursasDatabase
        """
        super().__init__(database, 'runner_time_dist.csv')

    def create_statistics_table(self):
        """
        Create the statistics table that can be used to make the figure.

        :return: The statistics to be cached.
        :rtype: pandas.DataFrame
        """

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
        """
        Generate the figure from the cached statistics.

        :return: The resulting figure.
        :rtype: plotly.graph_objects.Figure
        """

        #TODO: Unnamed: 0 column name
        data = self.get_statistics()

        #TODO: Hover doesn't work well for this plot
        fig = px.bar(data, x='Time', y=['Men', 'Women'])

        step_size = 1 #TODO: This assumes step size
        hist_group_boundries = np.arange(data['Time'].min(), data['Time'].max()+step_size, step_size)

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

        return fig

class OverallRunAmountsPlot(CachedStatFigure):
    """ A histogram of the amount runners that have logged an amount of results. """
    def __init__(self, database):
        """
        Constructor

        :param database: The accessor for the Cursas database
        :type database: CursasDatabase
        """
        super().__init__(database, 'overall_run_amounts.csv')

    def create_statistics_table(self):
        """
        Create the statistics table that can be used to make the figure.

        :return: The statistics to be cached.
        :rtype: pandas.DataFrame
        """
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
        """
        Generate the figure from the cached statistics.

        :return: The resulting figure.
        :rtype: plotly.graph_objects.Figure
        """

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
    """ The geometric mean of attendance through the lifespan of each event. """
    def __init__(self, database):
        """
        Constructor

        :param database: The accessor for the Cursas database
        :type database: CursasDatabase
        """
        super().__init__(database, 'average_attendance.csv')

    #TODO: Dropping nan means problem in dataset
    def create_statistics_table(self):
        """
        Create the statistics table that can be used to make the figure.

        :return: The statistics to be cached.
        :rtype: pandas.DataFrame
        """
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
        """
        Generate the figure from the cached statistics.

        :return: The resulting figure.
        :rtype: plotly.graph_objects.Figure
        """

        data = self.get_statistics()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=data['Days since start'],
            y=data['Mean Attendance'],
            name='Attendance'
            ),
            secondary_y=False
            )

        std_bound = np.array(
            (data['Mean Attendance'] + data['Std Attendance']/2).tolist() +
            (data['Mean Attendance'] - data['Std Attendance']/2).tolist()[::-1]
            )
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

class RunnerGroupsPlot(CachedStatFigure):
    """ The average and fastest times of the different age groups and sexes. """

    def __init__(self, database, data_file_name='runner_groups_avg_fast.csv'):
        """
        Constructor

        :param database: The accessor for the Cursas database
        :type database: CursasDatabase
        :param data_file_name: File name of where to save the data to in the database
        :type data_file_name: string
        """
        super().__init__(database, data_file_name)
        self.title = "Average and Fastest Times for Groups of Runners"

    def get_useable_results(self):
        """
        Get the part of the database needed to generate the statistics, the runner's times and attributes.

        :return: The relevant statistics.
        :rtype: pandas.DataFrame
        """
        results = self.database.get_all_results()[['Time', 'Athlete ID']]
        athletes = self.database.get_all_athletes()[['Sex', 'Age Group']]
        results = results.merge(athletes, how='outer', left_on='Athlete ID', right_index=True).dropna()
        del results['Athlete ID']
        results['Time'] = pd.to_numeric(results['Time'])

        return results

    def create_statistics_table(self):
        """
        Create the statistics table that can be used to make the figure.

        :return: The statistics to be cached.
        :rtype: pandas.DataFrame
        """
        stats_df = self.get_useable_results().groupby(['Sex', 'Age Group']).agg([
            'min',
            'median',
            'count'
            ])

        # Get rid of multi indices and columns
        stats_df = stats_df.reset_index()
        stats_df.columns = stats_df.columns.map(lambda x:x[0] if x[1] == '' else x[1])
        stats_df = stats_df.rename(columns={
            'min':'Fastest',
            'median':'Average',
            'count':'Number of Records'
            })

        #TODO: Is it faster to do this with some thing like apply?
        stats_df.loc[stats_df['Sex']=='M', 'Sex'] = 'Men'
        stats_df.loc[stats_df['Sex']=='W', 'Sex'] = 'Women'

        stats_df['Average']/=60
        stats_df['Fastest']/=60

        return stats_df

    def get_figure(self):
        """
        Generate the figure from the cached statistics.

        :return: The resulting figure.
        :rtype: plotly.graph_objects.Figure
        """

        data = self.get_statistics()
        fig = px.scatter(data, x='Average', y='Fastest', size='Number of Records', color='Sex', hover_data=['Age Group'])

        fig.update_layout(
            title=self.title,
            xaxis_title="Average Time (minutes)",
            yaxis_title="Fastest Time (minutes)",
        )

        return fig

class RunnerGroupsYearPlot(RunnerGroupsPlot):
    """ The average and fastest times of the different age groups and sexes over the last year. """
    def __init__(self, database):
        """
        Constructor

        :param database: The accessor for the Cursas database.
        :type database: CursasDatabase
        """
        super().__init__(database, data_file_name='runner_groups_avg_fast.csv')
        self.title = "Average and Fastest Times for Groups of Runners for the Last Year"

    def get_useable_results(self):
        """
        Get the part of the database needed to generate the statistics, the runner's times and attributes for the last year.

        :return: The relevant statistics.
        :rtype: pandas.DataFrame
        """
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
    """ A scatter plot for every different event with the average result time and attendance for the last year"""
    def __init__(self, database):
        """
        Constructor

        :param database: The accessor for the Cursas database
        :type database: CursasDatabase
        """
        super().__init__(database, 'event_avg_year.csv')

    def create_statistics_table(self):
        """
        Create the statistics table that can be used to make the figure.

        :return: The statistics to be cached.
        :rtype: pandas.DataFrame
        """
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
        return out_df\
                .merge(events, how='inner', left_on='Run ID', right_index=True)\
                .rename(columns={'Display Name':'Event Display Name'})

    def get_figure(self):
        """
        Generate the figure from the cached statistics.

        :return: The resulting figure.
        :rtype: plotly.graph_objects.Figure
        """

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
