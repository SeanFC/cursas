# Pull in the config values
#TODO: This doesn't respect sections
source <(grep = config/cursas.ini)

scp cursas $target/cursas
