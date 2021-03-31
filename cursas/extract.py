"""
Build an access a database of races, athletes and results
"""

# Standard library imports
import pickle as pkl
import datetime as dt

# External imports
from bs4 import BeautifulSoup
import requests
import pandas as pd
import scipy.stats as sps
import numpy as np

class CursasDatabase():
    """
    Races, athletes and results database

    For this database base the following definitions are used:
        * event: A race that is run somewhat regularly, usually from the same location.
        * athlete: A person that runs in races.
        * run_event: A particular race that was held by a particular event at a set time.
                     Run events involve athletes generate results.
        * result: A result time that an athlete got from participating in a run event.
    """
    def __init__(self, config):
        """
        Constructor

        :param config: All the configuration values for managing the database>
        :type config: CursasConfig
        """
        #TODO: Poor way to do this
        self.data_dir = config.data_dir
        self.sample_response_file_name = config.sample_response_file_name
        self.run_ids_file_name = config.run_ids_file_name
        self.full_table_file_name = config.full_table_file_name
        self.attendance_records_grab_file = self.data_dir + 'attendance_records.pkl'
        self.event_table_file = self.data_dir + 'events.csv'
        self.run_event_table_file = self.data_dir + 'run_events.csv'
        self.athlete_table_file = self.data_dir + 'athletes.csv'
        self.results_table_file = self.data_dir + 'results.csv'

    def full_database_refresh(self, pull_config, regrab=False):
        """
        Create the database from scratch

        :param pull_config: The configuration options for pulling the database from an external source as a
                            tuple of the external website address, the user agent to use to access the website.
        :type pull_config: 2-tuple
        :param regrab: Should the necessary parts of the database be downloaded (Default False).
        :type regrab: boolean
        """
        self.refresh_event_names(pull_config, regrab)

        #TODO: Lots of things aren't done here. This database refresh stuff should be put as its own class

    def refresh_event_names(self, pull_config, regrab=False):
        """
        Generate the list of event names.

        :param pull_config: The configuration options for pulling the database from an external source as a
                            tuple of the external website address, the user agent to use to access the website.
        :type pull_config: 2-tuple
        :param regrab: Should the event names be generated from an external source (Default False).
        :type regrab: boolean
        """
        external_website_db, user_agent = pull_config

        # If we generating from an external source then download the necessary information.
        if regrab:
            response = requests.get(external_website_db+'/results/attendancerecords/', headers={'User-Agent': user_agent,})

            with open(self.attendance_records_grab_file, 'wb') as attendance_file:
                pkl.dump((response), attendance_file)

        # Pull up the raw data
        with open(self.attendance_records_grab_file, 'rb') as attendance_file:
            response = pkl.load(attendance_file)

        # Parse raw data to get a list of event names
        soup = BeautifulSoup(response.content, 'html.parser')
        main_table = soup.find('table', attrs={"id":"results"})
        table_rows = main_table.find_all('tr')[1:] # Skipping the first row because it is the column names
        event_names = [ row.find_all('td')[0].get_text() for row in table_rows ]
        event_name_table = pd.DataFrame(
                {'Display Name':event_names},
                index=list(map(lambda x:x.replace(' ','').lower(), event_names)) #TODO: We're assuming these ids are unique
                )

        # Remove junior races
        junior_rows = event_name_table.apply(lambda row: 'junior' in row['Display Name'], axis=1)
        event_name_table = event_name_table[~junior_rows]

        # Remove any duplicates, often present in the data source
        event_name_table = event_name_table.drop_duplicates()

        # Save results
        event_name_table.to_csv(self.event_table_file)

    ## Getters
    def get_all_events(self):
        """
        Get the full events table.

        :return: The full events table.
        :rtype: panads.DataFrame
        """
        return pd.read_csv(self.event_table_file, index_col=0)

    def get_all_event_ids(self):
        """
        Get all the event ids.

        :return: All the event ids.
        :rtype: list
        """
        return pd.read_csv(self.event_table_file, index_col=0).index.tolist()

    def get_all_athletes(self):
        """
        Get the athletes table.

        :return: The full athletes table.
        :rtype: panads.DataFrame
        """
        return pd.read_csv(self.athlete_table_file, index_col=0)

    def get_all_run_events(self):
        """
        Get the run events table.

        :return: The full run events table.
        :rtype: panads.DataFrame
        """
        return pd.read_csv(self.run_event_table_file, index_col=0, parse_dates=['Date'])

    def get_first_run_events(self):
        """
        Get the first run event for every event.

        :return: A table with columns of the event id and the date of the first run event from that event.
        :rtype: panads.DataFrame
        """
        return self.get_all_run_events().groupby('Run ID')['Date'].min()

    def get_all_results(self):
        """
        Get the results table.

        :return: The full results table.
        :rtype: panads.DataFrame
        """
        return pd.read_csv(self.results_table_file, index_col=0)

    ## Setters
    def set_run_events(self, run_events_table):
        """
        Set all the run events in the database.

        :param run_events_table: The new run events to use with columns of 'Run ID' and 'Date'.
        :type run_events_table: pandas.DataFrame
        """
        run_events_table.to_csv(self.run_event_table_file)

    def set_athletes(self, athletes):
        """
        Set all the athletes in the database.

        :param athletes: The new athletes to use with columns of 'Sex' and 'Age Group'.
        :type athletes: pandas.DataFrame
        """
        athletes.to_csv(self.athlete_table_file)

    def set_results(self, results):
        """
        Set all the results in the database.

        :param results: The new results to use with columns of 'Athlete ID', 'Event ID' and 'Time'.
        :type results: pandas.DataFrame
        """
        results.to_csv(self.results_table_file)

# Note: An alternative approach here is to extend CursasDatabase but until the simulator needs find grained access to
# how the database is access the simulator and database should be kept separate.
class SimulateCursasDatabase():
    """
    Simulate data to put in a CursasDatabase.
    """

    def __init__(self):
        """
        Constructor
        """
        self.probability_of_cancelled_run_event = 1/50
        self.first_run_date = dt.date(year=2004, month=10, day=23)
        self.last_run_date = dt.date(year=2020, month=2, day=22)
        self.number_of_athletes = 142367
        self.possible_age_groups = ['18-19', *[f'{x}-{x+4}' for x in np.arange(20, 100, 5)], '100+']

    def generate_simulated_run_events(self, database):
        """
        Generate a simulated set of run events based on a list of events.

        :param database: The database to create run events for.
        :type database: CursasDatabase

        :return: The simulated run events
        :rtype: pandas.DataFrame
        """
        all_event_ids = database.get_all_event_ids()

        possible_start_weeks = (self.last_run_date - self.first_run_date).days/7
        # In units of days since the first run date
        start_date_distrubtion = np.floor(sps.beta(5, 1).rvs(size=len(all_event_ids))*possible_start_weeks)*7

        run_events_table = pd.DataFrame([], columns=['Run ID', 'Date'])

        #TODO: This whole loop is slow as a new pandas df is needed every time and we're doing lots of list
        #      comprehension on datetimes.
        for idx, event_id in enumerate(database.get_all_event_ids()):
            event_start_date = self.first_run_date + dt.timedelta(days=start_date_distrubtion[idx])

            possible_days_since_start = np.arange(0, (self.last_run_date - event_start_date).days, 7)

            # Simulate some cancellations
            cancelled_run_events = sps.bernoulli.rvs(self.probability_of_cancelled_run_event, size=len(possible_days_since_start))
            cancelled_run_events = cancelled_run_events.astype(bool)
            possible_days_since_start = possible_days_since_start[~cancelled_run_events]

            run_events_table = run_events_table.append([
                {
                    'Run ID':event_id,
                    'Date':event_start_date + dt.timedelta(days=int(days_since_start))
                    } for days_since_start in possible_days_since_start
                ])
        run_events_table.reset_index(drop=True, inplace=True)

        return run_events_table

    # Note: It's assumed for simplicity that athletes don't change over time (in terms of sex, age etc.) however,
    #       this is not the case.
    def generate_simulated_athletes(self):
        """
        Generate a simulated set of athletes.

        :return: The simulated athletes with columns 'Sex' and 'Age Group'.
        :rtype: pandas.DataFrame
        """
        num_age_groups = len(self.possible_age_groups)

        #TODO: Holding these items as strings is inefficient
        sexes = list(map(lambda x:'M' if x else 'W',sps.bernoulli.rvs(0.5, size=self.number_of_athletes)))
        age_groups = list(map(
            lambda x:self.possible_age_groups[np.min((x, num_age_groups))],
            sps.poisson.rvs(3, size=self.number_of_athletes)
            ))

        return pd.DataFrame({'Sex':sexes, 'Age Group':age_groups})

    # The approach here is to simulate each athlete's running 'career' one by one and then add any
    # additional information to each race as needed
    def generate_simulated_results(self, database):
        """
        Create random results for athletes running at run events.
        Currently athletes have a 'home event' this is the only event they run at.
        Each athlete's run times are drawn randomly from a distribution which has a mean which is based on their
        age, sex and random noise.

        :param database: The database to generate the simulated results from.
        :type database: CursasDatabase

        :return: The simulated results.
        :rtype: pandas.DataFrame
        """
        athletes = database.get_all_athletes()
        event_ids = database.get_all_event_ids()

        # Give everyone a 'home event' where they usually run at.
        # We shuffle the events since the distribution is based on their order (which is often alphabetical)
        np.random.shuffle(event_ids)
        home_events = np.array(event_ids)[(sps.beta(0.5, 0.5).rvs(size=len(athletes))*len(event_ids)).astype(int)]

        events_first_run_event = database.get_first_run_events()

        #TODO: Awkward date handling here
        max_career_lengths_weeks = np.floor(
                np.array([
                    (self.last_run_date - events_first_run_event.loc[home_event].to_pydatetime().date()).days/7
                    for home_event in home_events
                    ])
                )
        career_lengths_weeks = (
                max_career_lengths_weeks*sps.beta(1,3).rvs(len(max_career_lengths_weeks))
                ).astype(int) #TODO: Nothing to stop this being 0
        career_start_from_home_start_weeks = (
                (max_career_lengths_weeks - career_lengths_weeks)*sps.beta(5,2).rvs(len(career_lengths_weeks))
                ).astype(int)

        all_run_events = database.get_all_run_events()
        held_results = {'Athlete ID':[], 'Event ID':[], 'Time':[]} #TODO: Would be nice to preregister the size of these

        #TODO: This loop is quite inefficient
        for (athlete_id, athlete_data), home_event, career_start, career_length in zip(
                athletes.iterrows(), home_events, career_start_from_home_start_weeks, career_lengths_weeks
                ):
            print(athlete_id)
            #TODO: Could do this out of loop
            mean_run_time = (
                    28 +
                    3.5 * (-1 if athlete_data['Sex'] == 'M' else 1) +
                    self.possible_age_groups.index(athlete_data['Age Group']) - 3
                    )*60

            possible_events = all_run_events[
                    (all_run_events['Run ID'] == home_event) &
                    (all_run_events['Date'] > events_first_run_event[home_event] + dt.timedelta(days=int(career_start)*7))
                    ]
            possible_indicies = np.arange(0, len(possible_events))
            np.random.shuffle(possible_indicies)
            possible_events = possible_events.iloc[
                    possible_indicies[np.arange(0, np.min((career_length, len(possible_events))))]
                    ].sort_values('Date')

            #TODO: Add in some travelling runs here

            held_results['Athlete ID'].extend([athlete_id]*len(possible_events))
            held_results['Event ID'].extend(possible_events.index.tolist())
            #TODO: Poor model for run times. Also no sanity checks (e.g. what about super fast runs? or negative times?)
            held_results['Time'].extend(sps.norm(mean_run_time, 60).rvs(len(possible_events)).astype(int))

        return pd.DataFrame(held_results)

    def simulate_full_cursas_database(self, database):
        """
        Simulate a full Cursas dataset and put it into a given database.

        :param database: The database to generate the simulated results for.
        :type database: CursasDatabase
        """
        database.set_run_events(self.generate_simulated_run_events(database))
        database.set_athletes(self.generate_simulated_athletes())
        database.set_results(self.generate_simulated_results(database))

def pull_db(conf):
    """
    Recreate the full Cursas database.

    :param conf: The configuration to use for the database.
    :type conf: CursasConfig
    """
    database = CursasDatabase(conf)
    database.full_database_refresh((conf.external_website_db, conf.user_agent), regrab=False)
    SimulateCursasDatabase().simulate_full_cursas_database(database)
