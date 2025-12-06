remote_server="sufe197"
remote_dir="~/ml-framework"

ssh $remote_server "mkdir -p $remote_dir"

rsync -avcz --exclude='.git/' --exclude={'*.pyc','.venv/','output/*','.git/'} ./ "$remote_server:$remote_dir" --delete
echo 
rsync -avcz --exclude='.git/' --exclude={'*.pyc','.venv/'} "$remote_server:$remote_dir/" ./ --delete