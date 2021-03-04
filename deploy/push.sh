# exit when any command fails
set -e

# Pull in the config values
#TODO: This doesn't respect sections
source <(grep = config/cursas.ini)

# Build the systemd service script
cp deploy/cursas.service /tmp/cursas.service
echo "User=$user
WorkingDirectory=$deploy_path
ExecStart=/bin/bash $deploy_path/deploy/serv_run" >> /tmp/cursas.service

# Push the database
rsync -av -e "ssh -p $port" out/ "$target":"$deploy_db_path"

# Push the systemd service script
rsync -av -e "ssh -p $port" /tmp/cursas.service "$target":"$deploy_db_path"

# Push any new code
git push origin master

#TODO: These sudos are not great
ssh -p "$port" "$target" "\
    cd $deploy_path; \
    git pull origin master; \
    sudo -S ln -s "$deploy_db_path"/cursas.service /etc/systemd/system; \
    sudo -S systemctl daemon-reload; \
    sudo -S systemctl restart cursas; \
    "
