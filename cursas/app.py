""" Build the Cursas application by displaying all the figures. """

# External imports
import dash_core_components as dcc
import dash_html_components as html

# Internal imports
from cursas.extract import CursasDatabase
from cursas.visualise import (
        OverallAttendanceEventsPlot,
        YearlyAverageTimePlot,
        EventAvgYear,
        AverageAttendancePlot,
        RunnerTimeDistributionPlot,
        RunnerGroupsYearPlot,
        OverallRunAmountsPlot,
        )

def get_navbar():
    """
    Create the website wide navigation bar in Dash.

    :return: The dash based navigation bar element.
    :rtype: dash_html_components.html.Ul
    """
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
    """
    Create the main overview tab for the website.

    :param database: The database to make all the relevant figures and numbers from.
    :type database: CursasDatabase

    :return: List of html elements for the main tab
    :rtype: List of dash elements.
    """
    return [dcc.Markdown('''
    # Explore Simulated Parkrun Data

    **[Parkrun](https://www.parkrun.com/)** is a series of free to enter 5 kilometre running races held weekly across the world.
    This website shows interactive analysis on a **[simulated](#simulated)** version of the finish times for these races.

    This website has no affiliation with Parkrun and so has no access to the real data however, if you think you could help change this please **[contact me](https://www.sfcleator.com/contact)**.

    The code for this project is available on this **[website](https://www.sfcleator.com/cgit/cgit.cgi/cursas/)** or on **[Github](https://github.com/SeanFC/cursas)**.

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
    """
    Create a tab to describe the statistics of an average event in the database.

    :param database: The database to make all the relevant figures and numbers from.
    :type database: CursasDatabase

    :return: List of html elements for the tab.
    :rtype: List of dash elements.
    """
    return [
            dcc.Markdown('''
            ## Statistics of Events

            An event describes the location, management team, etc. that runs races on most Saturdays.
            This page to compares all the different events against each other and shows what a typical event looks like statistically.

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
    """
    Create a tab to describe the statistics of an average athlete in the database.

    :param database: The database to make all the relevant figures and numbers from.
    :type database: CursasDatabase

    :return: List of html elements for the tab.
    :rtype: List of dash elements.
    """
    return [
            dcc.Markdown('''
            ## Statistics of Athletes

            This page explores the demographics and race results of all the different athletes.

            '''),
            dcc.Graph(figure=RunnerTimeDistributionPlot(database).get_figure()),
            #dcc.Graph(figure=RunnerGroupsPlot(database).get_figure()),
            dcc.Graph(figure=RunnerGroupsYearPlot(database).get_figure()),
            dcc.Markdown('''
            The average here is calculated as the median.
            This means 50% of run times were faster and 50% slower than this average.
            '''),
            dcc.Graph(figure=OverallRunAmountsPlot(database).get_figure()),
            #dcc.Graph(figure=plot_single_performance()), #TODO: Move to single runner lookup tab
            ]

def build_full_app(app, config):
    """
    Add all the necessary UI elements to a Dash application to build the full application.

    :param app: The application to add UI elements to.
    :type app: dash.Dash
    :param config: The configuration values of the project.
    :type config: CursasConfig

    :return: The application with all the UI elements added.
    :rtype: dash.Dash
    """
    database = CursasDatabase(config)

    #TODO: Tabs don't change colour on hover
    #TODO: Rather than setting these options for every tab it would probably be better to set some
    #      of them for the whole tabs structure
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
    """
    Build a dash application with just one figure on it for development purposes.

    :param app: The application to add UI elements to.
    :type app: dash.Dash
    :param config: The configuration values of the project.
    :type config: CursasConfig

    :return: The application with all the UI elements added.
    :rtype: dash.Dash
    """
    # Build figure and create any needed statistics
    dev_figure_class = RunnerGroupsYearPlot(
            CursasDatabase(config)
            )
    dev_figure_class.create_statistics()

    # Add figure to the layout
    app.layout = html.Div(className="grid-container", children=[
        dcc.Graph(figure=dev_figure_class.get_figure())
        ])
    return app
