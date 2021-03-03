# Pull in the config values
#TODO: This doesn't respect sections
source <(grep = config/cursas.ini)

rsync -a cursas -e "ssh -p $port" "$target/cursas"
rsync -a serv_run -e "ssh -p $port" "$target"
