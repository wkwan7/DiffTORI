import time
import torch
import click
import socket
from chester.run_exp import run_experiment_lite, VariantGenerator
from src.train_run_task import run_task
import pdb

@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = '7.19-newtask'

    vg = VariantGenerator()

    ### For each parameter, you can also change to run multiple values, e.g. 
    # vg.add("modality", ["pixels", "state"])
    # vg.add("task", ["reacher-easy", "cheetah-run"])
    # vg.add("seed", [100, 200, 300])
    # this will launch 2 x 2 x 3 = 12 experiments automatically. 

    # vg.add("mode", ["back_plan"])
    # vg.add("task", ["finger-turn-hard", "fish-swim"])
    # vg.add("modality", ["state"])
    # vg.add("device", ["cuda"])
    # vg.add("seed", [1,2,3])
    # vg.add("max_iterations", [50])
    # vg.add("step_size", [0.001])
    # vg.add("update_from_buffer", [True])
    # vg.add("frequency", [10,20,50,100,200,500])
    # vg.add("grad_clip_norm_a_buffer", [1])
    # vg.add("damping", [0.001])

    vg.add("mode", ["default"]) #theseus_forward vg.add("mode", ["back_plan"])
    #vg.add("task", ["metaworld-assembly"])
    #vg.add("task", ["finger-turn-hard", "fish-swim", "acrobot-swingup", "hopper-hop", "quadruped-walk"])
    #vg.add("task", ["dog-run","dog-trot","dog-walk","humanoid-run","humanoid-stand","humanoid-walk"])
    #vg.add("task", ["walker-walk","walker-run","walker-stand","cup-catch"])
    vg.add("task", ["cartpole-balance-sparse"])
    # cartpole-balance cartpole-balance-sparse cartpole-swingup cartpole-swingup-sparse hopper-stand hopper-hop reacher-hard
    vg.add("modality", ["pixels"])
    vg.add("device", ["cpu"])

    vg.add("seed", [0,1,2])
    # vg.add("max_iterations", [50])
    # vg.add("step_size", [0.001])
    # vg.add("update_from_buffer", [False])
    # vg.add("damping", [0.001])

    # forward done: quadruped-walk, quadruped-run, finger-turn-hard , finger spin


    if debug:
        exp_prefix += '_debug'

    print('Number of configurations: ', len(vg.variants()))
    print("exp_prefix: ", exp_prefix)

    hostname = socket.gethostname()
    gpu_num = torch.cuda.device_count()

    variations = set(vg.variations())
    task_per_gpu = 1
    all_vvs = vg.variants()
    slurm_nums = len(all_vvs) // task_per_gpu
    if len(all_vvs) % task_per_gpu != 0:
        slurm_nums += 1
    
    for vvs in all_vvs:
        vvs['exp_name'] = vvs['mode']

        # vvs['exp_name'] = vvs['mode'] + '_' + 'freq=' + str(vvs['frequency']) + '_' + \
        #  'grad=' + str(vvs['grad_clip_norm_a_buffer']) + '_' + 'damp=' + str(vvs['damping']) + \
        #  '_' + str(vvs['max_iterations']) + '_' + str(vvs['step_size'])
        
        # vvs['exp_name'] = vvs['mode'] + '_' + 'damp=' + str(vvs['damping']) + \
        #  '_' + str(vvs['max_iterations']) + '_' + str(vvs['step_size']) #+ '_' + 'ar=4' #+ '_' +'zeroinit'

    sub_process_popens = []
    for idx in range(slurm_nums):
        beg = idx * task_per_gpu
        end = min((idx+1) * task_per_gpu, len(all_vvs))
        vvs = all_vvs[beg:end]
        while len(sub_process_popens) >= 10:
            sub_process_popens = [x for x in sub_process_popens if x.poll() is None]
            time.sleep(10)
        if mode in ['seuss', 'autobot']:
            if idx == 0:
                compile_script = None  
                wait_compile = 0
            else:
                compile_script = None
                wait_compile = 0  
        elif mode == 'ec2':
            compile_script = 'compile_1.0.sh'
            wait_compile = None
        else:
            compile_script = wait_compile = None
        if hostname.startswith('autobot') and gpu_num > 0:
            env_var = {'CUDA_VISIBLE_DEVICES': str(idx % gpu_num)}
        else:
            env_var = None
        cur_popen = run_experiment_lite(
            stub_method_call=run_task,
            variants=vvs,
            # variant=vv,
            mode=mode,
            dry=dry,
            use_gpu=True,
            exp_prefix=exp_prefix,
            wait_subprocess=debug,
            compile_script=compile_script,
            wait_compile=wait_compile,
            env=env_var,
            variations=variations,
            task_per_gpu=task_per_gpu
        )
        if cur_popen is not None:
            sub_process_popens.append(cur_popen)
        if debug:
            break


if __name__ == '__main__':
    main()
