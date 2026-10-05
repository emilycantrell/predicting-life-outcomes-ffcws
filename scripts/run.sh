exp_id=$2
conf_path=$(readlink --canonicalize $1)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate million-monkeys
cd million-monkeys
exp_dir=$(dirname "$conf_path")
tar_file=$exp_dir".tar.gz"
echo "Tar file"$tar_file >>/dev/stderr
echo "Exp dir"$exp_dir >>/dev/stderr
echo $tar_file
python monkeys.py run $conf_path -e $2
python monkeys.py export-to-json $exp_dir $3
python monkeys.py verify-json-files $exp_dir
tar -zcvf $tar_file -P $exp_dir
