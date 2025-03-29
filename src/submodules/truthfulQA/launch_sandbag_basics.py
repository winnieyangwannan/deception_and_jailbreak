import os
import subprocess
import re
from os.path import join
import itertools
import argparse
# model_path = "meta-llama/Llama-3.1-8B"


def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument(
        "--model_path", 
        type=str,
        required=False, 
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Path to the model"
    )

    parser.add_argument(
        "--source_layer",
        type=int,
        required=False,
        default=14,
        help="layer to extract the steering vector",
    )
    parser.add_argument(
        "--target_layer",
        type=int,
        required=False,
        default=14,
        help="layer to steer the model",
    )
    parser.add_argument(
        "--steering_strength",
        type=int,
        required=False,
        default=1,
        help="coefficient to multiply the steering vector",
    )
    return parser.parse_args()

def run_subprocess_slurm(command):
    # Execute the sbatch command
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Print the output and error, if any
    print("command:", command)
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)

    # Get the job ID for linking dependencies
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if match:
        job_id = match.group(1)
    else:
        job_id = None

    return job_id


def main(model_path="meta-llama/Llama-3.2-1B-Instruct",
         source_layer=14, target_layer=14, steering_strength=1):
    model_name = os.path.basename(model_path)


    # job_name = "entity_source_" + model_name_source + "_target_" + model_name_target
    gpus_per_node = 1
    nodes = 1
    total_gpus = gpus_per_node * nodes

    job_name = f"basics_sandbag_" + model_name  + f"_{str(total_gpus)}"

    # get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(current_dir)
    os.chdir(current_dir)
    current_directory = os.getcwd()
    print("current_directory:", current_directory)


    print("model_name", model_name)

    slurm_cmd = f'''sbatch --account=genai_interns --qos=lowest \
                --job-name={job_name} --nodes={nodes} --gpus-per-node={gpus_per_node} \
                --mem-per-gpu=80G \
                --time=24:00:00 --output=logs/{job_name}.log \
                --wrap="srun accelerate launch \
                --num_processes={total_gpus} \
                --num_machines={nodes} \
                --machine_rank=\$SLURM_PROCID \
                --main_process_ip=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1) \
                --main_process_port=29500 \
                run_sandbag_basics.py \
                --model_path={model_path}"
                '''
    job_id = run_subprocess_slurm(slurm_cmd)
    print("job_id:", job_id)



if __name__ == "__main__":
    args = parse_arguments()
    main(model_path=args.model_path,
         source_layer=args.source_layer, 
         target_layer=args.target_layer,
         steering_strength=args.steering_strength)
