!/bin/bash

log_folder="${pwd}/log_output/$1_output"

if [ ! -x $log_folder ]; then
	mkdir $log_folder
fi

run_times=$2
IFS=" "
datasets=($@)
unset datasets[0]
unset datasets[1]

echo $datasets

for (( i=1; i <= run_times; i++))
do
  for dataset in ${datasets[@]}
  do
    if [[ $dataset =~ "/" ]];then   # 针对new-data/co-az类型创建一个目录new-data
      par_dir=(${dataset//// }[0])
      new_dir=${log_folder}"/"$par_dir
      echo "Make dir: $new_dir"
      if [ ! -x "$new_dir" ]; then
        mkdir "$new_dir"
      fi
    fi
    dataset_dir="/home/chengfeng/autograph/public/$dataset"
    cur_time="`date +%Y-%m-%d-%H-%M-%S`"
    log_file="$log_folder/$dataset-$cur_time.log"
    python_command="python run_local_test.py --dataset_dir=$dataset_dir 2>&1"
    log_command="tee -i $log_file"
    echo "Current time: $cur_time"
    echo "Run command: $python_command"
    echo "Log info into file: $log_file"
    eval "$python_command | $log_command"
  done
done
