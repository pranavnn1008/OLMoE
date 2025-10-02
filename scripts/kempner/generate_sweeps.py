import itertools

def generate_commands():
    # --- Define the sweep space ---
    learning_rates =  [1e-2] #, 1e-3, 3e-3]
    batch_sizes = [128]

    # --- Define constant paths and base command ---
    # !!! IMPORTANT: Update these paths to match your environment !!!
    olmo_train_script = "/n/holylfs06/LABS/sham_lab/Lab/pranav/OLMo/scripts/train.py"
    base_config_file = "/n/holylfs06/LABS/sham_lab/Lab/pranav/OLMo/configs/official/OLMoE-sweep-config.yaml" # Make sure this path is correct
    
    base_command = f"torchrun --nproc_per_node=1 {olmo_train_script} {base_config_file}"

    # --- Generate all combinations and create commands ---
    commands = []
    for lr, bs in itertools.product(learning_rates, batch_sizes):
        #
        # === KEY CHANGE IS HERE ===
        # Create a unique, descriptive run_name for each experiment.
        # The f-string formatting `{lr:.1e}` ensures scientific notation is clean (e.g., 1.0e-04).
        #
        # run_name = f"olmoe_no_z_loss_lr_{lr:.1e}_bs_{bs}"
        run_name = f"olmoe_baseline_lr_{lr:.1e}_bs_{bs}"
        # max_duration = int(14661811200 / (1024 * bs))  # Adjust max_duration based on batch size
        max_duration = int(6628812800 / (1024 * bs))  # Adjust max_duration based on batch size

        # Build the override arguments. By setting `run_name` here, we automatically
        # update `wandb.name` and `save_folder` thanks to the YAML config.
        overrides = [
            f"run_name={run_name}",
            f"optimizer.learning_rate={lr}",
            f"global_train_batch_size={bs}",
            f"max_duration={max_duration}",
        ]
        
        full_command = f"{base_command} " + " ".join(overrides)
        commands.append(full_command)

    # --- Write commands to a file ---
    output_file = "/n/netscratch/sham_lab/Lab/pranavajitnair/moe_sweeps_files/sweep_commands.txt"
    with open(output_file, "w") as f:
        for command in commands:
            f.write(command + "\n")
            
    print(f"✅ Generated {len(commands)} commands in '{output_file}'.")
    print(f"Next, update your Slurm script's --array directive to: --array=1-{len(commands)}")
    print("\nExample command from the file:")
    print(commands[0])

if __name__ == "__main__":
    generate_commands()