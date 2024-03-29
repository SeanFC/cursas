# Cursas 

Generate simulated Parkrun data, visualise it and then make predictions. 
The hope is to do this for the real Parkrun data.

This repository includes tools for generating the website found [here](https://www.sfcleator.com/cursas) and is part of the Sudorn project.

Name from 'cursus', Latin for running. 

## Install and Run

Install with 
```
pip install -e . 
```
from inside the main directory.
Then run with
```
python -m cursas
```
to generate the full website.

### Alternate running modes

For development a site with a single graph can be run with 
```
python -m cursas dev 
```
and the database can be create/pulled with
```
python -m cursas pull
```

## Deploy

Deployment requires a server running with:

* a user that can run uwsgi and Cursas
* an appropriate cursas.ini file
* a Cursas repository set up with it's origin to pull from the same repository was being pushed to

The deploy command is then
```
./deploy/push.sh
```

# TODO

* Deploy still requires some server side setup
* Lots of confusing config options
* Need to use assets of wider website
* Add a weekly update - how was this weeks event, how was this weeks park run overall (different demographics?), how did individuals do
* Covid analysis
* Install as command line program (python -m cursas -> ./cursas)
* Plotting
    * Add record times (world record and PR record) to relevant plots
    * Sex and gender naming mixing
    * Add a demographics plot (the distributions of age and sex)
    * Red blue colour scheme doesn't really go with the rest of the website. Need a different accent colour than red.
* Clean code and TODO notes in code files.
* Finish initial linting
* Write tests for database simulator

## Contact
Maintained by Sean F. Cleator

Email: seancleator@hotmail.co.uk 
