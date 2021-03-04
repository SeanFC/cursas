# exit when any command fails
set -e

# Pull in the config values
#TODO: This doesn't respect sections
source <(grep = config/cursas.ini)

git push origin master

ssh -p "$port" "$target" "\
    cd $deploy_path; \
    git pull origin master; \
    sudo -S systemctl daemon-reload; \
    sudo -S systemctl restart cursas; \
    "
