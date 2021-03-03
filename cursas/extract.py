from os import listdir
from os.path import isfile, join
import requests
import pickle as pkl
from bs4 import BeautifulSoup

import datetime as dt

import re

from cursas.config import *

class RunEvent():
    def __init__(self, header, run_name, event_id):
        self.run_name = run_name
        self.event_id = event_id
        date_string = header.find('h3').text.split(" ")[0].split("/")
        self.date = dt.date(year=int(date_string[2]), month=int(date_string[1]), day=int(date_string[0]))

class ResultEntry():
    athelete_link_preamble_length = len("athletehistory?athleteNumber=")

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
                self.athlete_id = ath_link_item['href'][ResultEntry.athelete_link_preamble_length:]
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

def plot_single_run(race_entries):
    time_increase = [ row.time - row.pb_time for row in race_entries if row.athlete_id != -1 and row.time != row.pb_time ]
    times = np.array([ row.time for row in race_entries if row.athlete_id != -1 ])
    #plt.hist(time_increase, bins=np.arange(0, 1400, 10))
    plt.hist(times/60, bins=40)#, bins=np.arange(0, 1400, 10))
    plt.show()

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

if __name__ == "__main__":
    #event_ids = parse_all_run_ids()
    #grab_all_runs_from_ids(event_ids)

    generate_full_run_table('eastville')
