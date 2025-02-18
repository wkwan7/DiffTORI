# bash scripts/gen_demonstration_metaworld.sh assembly


# full metaworld task list:


cd third_party/BEE

task_name=${1}


# if not use script policy, then use the expert policy trained with BAC.
use_script_policy=True

export CUDA_VISIBLE_DEVICES=0
python gen_demonstration_expert.py --env_name=${task_name} \
            --num_episodes 10 \
            --root_dir "../../DiffTORI/data/" \
            --use_point_crop True \
            --num_points 512 \
            --use_script_policy $use_script_policy
