#/usr/bin/env sh


data_dir=${1-./data}
fls=${2-test}
num_workers=${3-20}

python \
    ./src/gen_seq.py \
        --file-path "${data_dir}"/"${fls}".jsonl \
        --output-path "${data_dir}"/"${fls}".csv \
        --num-workers "${num_workers}" \


# if you don't want to install `jq`, just create a `grid.json` file where every
# row is a `grid_info` (unique):
#   {"width":1080,"height":667,"keys":[...],"grid_name":"default"}
#   {"width":1080,"height":667,"keys":[...],"grid_name":"extra"}
if [ "${fls}" = "test" ]; then
    jq -c '.[].grid' "${data_dir}"/test.jsonl  | sort -u > "${data_dir}"/grid.json
fi
