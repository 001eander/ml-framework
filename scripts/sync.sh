remote_server="sufe197"
remote_dir="~/demo-ml/"


echo "Sync local to remote"
ssh $remote_server "mkdir -p $remote_dir"
rsync -avczq --delete --exclude='.gitkeep' ./conf/ "$remote_server:$remote_dir/conf/"
# rsync -avczq --delete --exclude='.gitkeep' ./exp/ "$remote_server:$remote_dir/exp/"
rsync -avczq --delete ./scripts/ "$remote_server:$remote_dir/scripts/"
rsync -avczq --delete ./src/ "$remote_server:$remote_dir/src/"
rsync -avcq ./pyproject.toml "$remote_server:$remote_dir/pyproject.toml"
rsync -avcq ./uv.lock "$remote_server:$remote_dir/uv.lock"

echo "Sync remote to local"
ssh $remote_server "mkdir -p $remote_dir/output"
rsync -avczq --delete --exclude='.gitkeep' "$remote_server:$remote_dir/output/" ./output/