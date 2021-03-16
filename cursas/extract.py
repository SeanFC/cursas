from os import listdir
from os.path import isfile, join
import requests
import pickle as pkl
from bs4 import BeautifulSoup

import datetime as dt

import re
import pandas as pd
import scipy.stats as sps
import numpy as np

class RunEvent():
    def __init__(self, header, run_name, event_id):
        self.run_name = run_name
        self.event_id = event_id
        date_string = header.find('h3').text.split(" ")[0].split("/")
        self.date = dt.date(year=int(date_string[2]), month=int(date_string[1]), day=int(date_string[0]))

class ResultEntry():
    athlete_link_preamble_length = len("athletehistory?athleteNumber=")

    def __init__(self, row_data, run_name, event_id):
        #print(row_data.prettify())
        self.run_name = run_name
        self.event_id = event_id
        self.achievement = row_data['data-achievement']
        self.age_grade = row_data['data-agegrade']
        self.age_group = row_data['data-agegroup']
        self.club = row_data['data-club']
        self.name = row_data['data-name']
        self.position = row_data['data-position']
        self.runs = row_data['data-runs']
        self.athlete_id = -1
        self.time = -1
        self.pb_time = -1

        detailed_name = row_data.find("td", attrs={"class":"Results-table-td Results-table-td--name"})
        if detailed_name is not None:
            ath_link_item = detailed_name.find("a")
            if ath_link_item is not None:
                self.athlete_id = ath_link_item['href'][ResultEntry.athlete_link_preamble_length:]
                try:
                    self.athlete_id = int(self.athlete_id)
                except ValueError: 
                    self.athlete_id = -2

        detailed_time = row_data.find("td", attrs={"class":re.compile("Results-table-td Results-table-td--time*")})
        if self.athlete_id != -1:
            if detailed_time is not None:
                #TODO: Repeated time splitting stuff
                time_section = detailed_time.find("div", attrs={"class":"compact"})
                if time_section is not None:
                    self.time = time_section.string.split(":")
                    self.time = int(self.time[0])*60 + int(self.time[1])
                time_section = detailed_time.find("div", attrs={"class":"detailed"})
                if time_section is not None:
                    for cur_str in time_section.strings:
                        if ':' in cur_str:
                            self.pb_time = cur_str.split(":")
                            self.pb_time = int(self.pb_time[0])*60 + int(self.pb_time[1])

         #TODO: There is more data in these tables (such as if the athlete has done over 100 runs, and gender position)

    def __repr__(self):
        return "<ResultEntry Object, athl id={}>".format(self.athlete_id)

def grab_latest_run():
    headers = {'User-Agent': user_agent,}
    response = requests.get(external_website_db+'/eastville/results/latestresults/', headers=headers)

    with open(sample_response_file_name, 'wb') as f:  
        pkl.dump((response), f)

def parse_single_run(file_name, run_name, event_id):
    with open(file_name, 'rb') as f:  
        response = pkl.load(f)
    soup = BeautifulSoup(response.content, 'html.parser')

    #TODO: Also scrape volunteer data
    main_table = soup.find('table', attrs={"class":"Results-table Results-table--compact js-ResultsTable"})
    table_rows = main_table.find_all('tr', attrs={"class":"Results-table-row"})

    header = soup.find('div', attrs={"class":"Results-header"})
    
    return RunEvent(header, run_name, event_id), [ ResultEntry(row, run_name, event_id) for row in table_rows ]


def grab_all_run_ids():
    headers = {'User-Agent': user_agent,}
    response = requests.get(external_website_db+'/eastville/results/eventhistory/', headers=headers)

    with open(run_ids_file_name, 'wb') as f:  
        pkl.dump((response), f)

def parse_all_run_ids():
    with open(run_ids_file_name, 'rb') as f:  
        sample_response = pkl.load(f)
    soup = BeautifulSoup(sample_response.content, 'html.parser')

    main_table = soup.find('table', attrs={"id":"results"}).find("tbody")
    table_rows = main_table.find_all('tr')#, attrs={"class":" even"})

    return [ r.find("td").find("a").string for r in table_rows ]

def grab_all_runs_from_ids(ids):
    headers = {'User-Agent': user_agent,}
    only_files = [f.split('.')[0] for f in listdir('out/eastville') if isfile(join('out/eastville', f))]
    target_ids = list(set(ids) - set(only_files))#[:2]

    random.shuffle(target_ids)
    for cur_id in target_ids:
        print(cur_id)
        response = requests.get(external_website_db+'/eastville/results/weeklyresults/?runSeqNumber={}'.format(cur_id), headers=headers)

        with open('out/eastville/{}.pkl'.format(cur_id), 'wb') as f:  
            pkl.dump((response), f)

        time.sleep(np.random.rand()*20)

def generate_full_run_table(run_name):
    only_files = [f.split('.')[0] for f in listdir('out/eastville') if isfile(join('out/eastville', f))]

    all_rows = []
    all_events = []

    for cur_id in only_files:
        print(cur_id)
        event_entry, row_entries = parse_single_run('out/{}/{}.pkl'.format(run_name, cur_id), run_name, cur_id)

        all_events.append(event_entry)
        for r in row_entries:
            all_rows.append(r)

    with open(full_table_file_name, 'wb') as f:  
        pkl.dump((all_events, all_rows), f)

def pull_db(conf):
    db = CursasDatabase(conf)
    db.full_database_refresh((conf.external_website_db, conf.user_agent), regrab=False)

class CursasDatabase():
    def __init__(self, config):
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
        #self.refresh_event_names(pull_config, regrab)

        #TODO: DB elements being generated
        #self.generate_simulated_run_events() 
        #self.generate_simulated_athletes() 
        self.generate_simulated_results()

    def refresh_event_names(self, pull_config, regrab=False):
        external_website_db, user_agent = pull_config
    
        if regrab:
            response = requests.get(external_website_db+'/results/attendancerecords/', headers={'User-Agent': user_agent,})

            with open(self.attendance_records_grab_file, 'wb') as f:  
                pkl.dump((response), f)

        with open(self.attendance_records_grab_file, 'rb') as f:  
            response = pkl.load(f)

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

        event_name_table.to_csv(self.event_table_file)

    def get_all_event_ids(self):
        return pd.read_csv(self.event_table_file, index_col=0).index.tolist()

    def get_all_athletes(self):
        return pd.read_csv(self.athlete_table_file, index_col=0)

    def get_all_run_events(self):
        return pd.read_csv(self.run_event_table_file, index_col=0, parse_dates=['Date'])

    def get_first_run_events(self):
        run_events = self.get_all_run_events()
        return run_events.groupby('Run ID')['Date'].min()

    def get_all_results(self):
        return pd.read_csv(self.results_table_file, index_col=0)

    #TODO: All this simulated stuff shouldn't be in the database class. It should interface with the db class using member functions
    def generate_simulated_run_events(self):
        probability_of_cancelled_run_event = 1/50
        first_run_date = dt.date(year=2004, month=10, day=23)
        last_run_date = dt.date(year=2020, month=2, day=22)

        all_event_ids = self.get_all_event_ids()

        possible_start_weeks = (last_run_date - first_run_date).days/7
        start_date_distrubtion = np.floor(sps.beta(5, 1).rvs(size=len(all_event_ids))*possible_start_weeks)*7 # In units of days since the first run date

        run_events_table = pd.DataFrame([], columns=['Run ID', 'Date'])

        #TODO: This whole loop is slow as a new pandas df is needed every time and we're doing lots of list comprehension on datetimes 
        for idx, event_id in enumerate(self.get_all_event_ids()):
            event_start_date = first_run_date + dt.timedelta(days=start_date_distrubtion[idx])

            possible_days_since_start = np.arange(0, (last_run_date - event_start_date).days, 7)

            # Simulate some cancellations
            cancelled_run_events = sps.bernoulli.rvs(probability_of_cancelled_run_event, size=len(possible_days_since_start))
            cancelled_run_events = cancelled_run_events.astype(bool)
            possible_days_since_start = possible_days_since_start[~cancelled_run_events]

            run_events_table = run_events_table.append(
                    [
                        {
                            'Run ID':event_id, 
                            'Date':event_start_date + dt.timedelta(days=int(days_since_start))
                            } for days_since_start in possible_days_since_start
                        ],
                    )

        run_events_table.to_csv(self.run_event_table_file)


    #TODO: The idea of having an athlete table doesn't really work since people can change age, gender and name
    def generate_simulated_athletes(self):
        number_of_athletes = 142367
        possible_age_groups = ['18-19', *[f'{x}-{x+4}' for x in np.arange(20, 100, 5)], '100+']
        num_age_groups = len(possible_age_groups)

        #TODO: Holding these items as strings is inefficient
        sexes = list(map(lambda x:'M' if x else 'W',sps.bernoulli.rvs(0.5, size=number_of_athletes)))
        age_groups = list(map(lambda x:possible_age_groups[np.min((x, num_age_groups))], sps.poisson.rvs(3, size=number_of_athletes)))

        df = pd.DataFrame({'Sex':sexes, 'Age Group':age_groups})
        df.to_csv(self.athlete_table_file)

    def generate_simulated_results(self):
        # The approach here is to simulate each athlete's running 'career' one by one and then add any additional information to each race as needed

        last_run_date = dt.date(year=2020, month=2, day=22) #TODO: This is copied from above
        possible_age_groups = ['18-19', *[f'{x}-{x+4}' for x in np.arange(20, 100, 5)], '100+'] #TODO: Copied from above

        athletes = self.get_all_athletes()#.iloc[:4]
        event_ids = self.get_all_event_ids()#[:4]

        np.random.shuffle(event_ids) # We shuffle the events so the distribution is based on their order (which is often alphabetical)
        home_events = np.array(event_ids)[(sps.beta(0.5, 0.5).rvs(size=len(athletes))*len(event_ids)).astype(int)]

        events_first_run_event = self.get_first_run_events()

        #TODO: Awkward date handling here
        max_career_lengths_weeks = np.floor(np.array([(last_run_date - events_first_run_event.loc[home_event].to_pydatetime().date()).days/7 for home_event in home_events]))
        career_lengths_weeks = (max_career_lengths_weeks*sps.beta(1,3).rvs(len(max_career_lengths_weeks))).astype(int) #TODO: Nothing to stop this being 0
        career_start_from_home_start_weeks = ((max_career_lengths_weeks - career_lengths_weeks)*sps.beta(5,2).rvs(len(career_lengths_weeks))).astype(int)

        all_run_events = self.get_all_run_events()
        held_results = {'Athlete ID':[], 'Event ID':[], 'Time':[]} #TODO: Would be nice to preregister the size of these
    
        #TODO: This loop is quite inefficient
        for (athlete_id, athlete_data), home_event, career_start, career_length in zip(athletes.iterrows(), home_events, career_start_from_home_start_weeks, career_lengths_weeks):
            print(athlete_id)
            #TODO: Could do this out of loop
            mean_run_time = (
                    28 + 
                    3.5 * (-1 if athlete_data['Sex'] == 'M' else 1) + 
                    possible_age_groups.index(athlete_data['Age Group']) - 3
                    )*60

            possible_events = all_run_events[
                    (all_run_events['Run ID'] == home_event) & 
                    (all_run_events['Date'] > events_first_run_event[home_event] + dt.timedelta(days=int(career_start)*7)) 
                    ]
            possible_indicies = np.arange(0, len(possible_events)) 
            np.random.shuffle(possible_indicies)
            possible_events = possible_events.iloc[possible_indicies[np.arange(0, np.min((career_length, len(possible_events))))]] .sort_values('Date')
            
            #TODO: Add in some travelling runs here

            held_results['Athlete ID'].extend([athlete_id]*len(possible_events))
            held_results['Event ID'].extend(possible_events.index.tolist())
            held_results['Time'].extend(sps.norm(mean_run_time, 60).rvs(len(possible_events)).astype(int)) #TODO: Poor model for run times. Also no sanity checks (e.g. what about super fast runs? or negative times?)

            #TODO; Making a double dataframe here?
            #results = results.append(pd.DataFrame({
            #    'Athlete ID': [athlete_id]*len(possible_events), 
            #    'Event ID': possible_events.index,
            #    'Time': sps.norm(mean_run_time, 60).rvs(len(possible_events)).astype(int) #TODO: Poor model for run times. Also no sanity checks (e.g. what about super fast runs? or negative times?)
            #    }), ignore_index=True)

        results = pd.DataFrame(held_results)#[], columns=['Athlete ID', 'Event ID', 'Time'])
        results.to_csv(self.results_table_file)

if __name__ == "__main__":
    #event_ids = parse_all_run_ids()
    #grab_all_runs_from_ids(event_ids)

    generate_full_run_table('eastville')
