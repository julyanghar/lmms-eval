import subprocess
import itertools
import os

run_names = [
    "no-img-text_filter-upper-0.9-lower-0-lr-2e-05-batch-12-beta-0.1-anchor-True-llava-v1.5-13b-0"
]


def eval_sft():
    master_ports = [60000,60001]
    processes = []
    i = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
    # for ls_factor_text_weight in ls_factor_text_weights:
    if i >= len(master_ports):
        raise ValueError(f"i={i} 超出 master_ports 长度 {len(master_ports)}")
    master_port = master_ports[i]
    i += 1
    # os.makedirs(output_dir, exist_ok=True)
    # base_model="llava-v1.5-7b"
    # base_model="llava-v1.5-13b"
    base_model="llava-v1.6-mistral-7b"
    model_name=f"liuhaotian/{base_model}"

    # tasks="pope_random,pope_pop,pope_adv"
    tasks="pope_random,pope_pop,pope_adv," \
        "mme,scienceqa,scienceqa_img,vizwiz_vqa"
    # tasks="amber"

    cmd = [
    # f"""python -m debugpy --connect 5679 -m accelerate.commands.launch \
    f"""python3 -m accelerate.commands.launch\
    --num_processes=6 \
    --main_process_port {master_port} \
    -m lmms_eval \
    --model llava \
    --model_args 'pretrained={model_name}' \
    --tasks {tasks} \
    --batch_size 1 \
    --output_path './output/' """
    ]

    p = subprocess.Popen(cmd, shell=True)
    p.cmd = cmd
    
    processes.append(p)
    print(f"当前进程数: {len(processes)}")
        # 如果已经启动了 2 个，就等待它们跑完
    if len(processes) == 1:
        for p in processes:
            p.wait()     # 等待当前这批都结束
            if p.returncode != 0:
                print(f"❌ Failed: {base_model} cmd={p.cmd}")
            else:
                print(f"✅ Finished: {base_model} cmd={p.cmd}")
        i = 0
        processes = []     # 清空进程列表，开始下一批
        


def eval():
    master_ports = [60121,60111]
    processes = []
    i = 0
    base_model="llava-v1.5-13b"
    model_name=f"liuhaotian/{base_model}"
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    for run_name in run_names:
        if i >= len(master_ports):
            raise ValueError(f"i={i} 超出 master_ports 长度 {len(master_ports)}")
        master_port = master_ports[i]
        i += 1
        # os.makedirs(output_dir, exist_ok=True)

        # pretrained 为checkpoint保存路径
        pretrained=f"/home/yilin/yilin-DPO/output/{base_model}/{run_name}"
        tasks="pope_random,pope_pop,pope_adv"
        # tasks="pope_random,pope_pop,pope_adv," \
        # "mme,object_hallucination,scienceqa,scienceqa_img,vizwiz_vqa"
        # tasks="amber"
        # tasks="mmvet"

        cmd = [
        # f"""python -m debugpy --connect 5679 -m accelerate.commands.launch \
        f"""python3 -m accelerate.commands.launch\
        --num_processes=1 \
        --main_process_port {master_port} \
        -m lmms_eval \
        --model llava \
        --model_args 'pretrained={pretrained},model_name={model_name},lora=True' \
        --tasks {tasks} \
        --batch_size 1 \
        --output_path './output/' \
        --wandb_log_samples \
        --wandb_args 'project=lmms-eval,job_type=eval,name={run_name}' """
        ]

        p = subprocess.Popen(cmd, shell=True)
        p.cmd = cmd
        
        processes.append(p)
        print(f"当前进程数: {len(processes)}")
            # 如果已经启动了 1 个，就等待它们跑完
        if len(processes) == 1:
            for p in processes:
                p.wait()     # 等待当前这批都结束
                if p.returncode != 0:
                    print(f"❌ Failed: {base_model} cmd={p.cmd}")
                else:
                    print(f"✅ Finished: {base_model} cmd={p.cmd}")
            i = 0
            processes = []     # 清空进程列表，开始下一批


    for p in processes:
        p.wait()     # 等待当前这批都结束
        if p.returncode != 0:
            print(f"❌ Failed: {base_model} cmd={p.cmd}")
        else:
            print(f"✅ Finished: {base_model} cmd={p.cmd}")
    processes = [] 



# eval()
eval_sft()