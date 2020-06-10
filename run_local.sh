#!/bin/bash

log_folder="./log_output"
if [ ! -x $log_folder ]; then
	mkdir $log_folder
fi

for arg in $*
do
  if [[ $arg =~ "/" ]];then   # 针对new-data/co-az类型创建一个目录new-data
    par_dir=(${arg//// }[0])
    new_dir=${log_folder}"/"$par_dir
    echo "Make dir: $new_dir"
    if [ ! -x "$new_dir" ]; then
      mkdir "$new_dir"
    fi
  fi
	dataset_dir="../public/$arg"
	cur_time="`date +%Y-%m-%d-%H-%M-%S`"
	log_file="$log_folder/$arg-$cur_time.log"
	python_command="python run_local_test.py --dataset_dir=$dataset_dir 2>&1"
	log_command="tee -i $log_file"
	echo "Current time: $cur_time"
	echo "Run command: $python_command"
	echo "Log info into file: $log_file"
	eval "$python_command | $log_command"
done
