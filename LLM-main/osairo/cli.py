import click
import sys
import re
import os
from .data_manager import load_csv
from .model_manager import train_model
from .active_learning import active_learning_cycle
from .config import DEFAULT_RESPONSES_FOLDER
from .knowledge_mode import knowledge_chat_session

def colorful_print(msg, color="white", bold=False):
    click.secho(msg, fg=color, bold=bold)

def greet_user():
    colorful_print("Welcome to osairo (All-purpose Molecular Modeling & Active Learning)!", color="bright_cyan", bold=True)
    colorful_print("Type 'help' for instructions, 'chat' for Q&A, or 'exit' to quit.", color="yellow")

def parse_feature_input(df, prompt_text):
    while True:
        raw = click.prompt(click.style(prompt_text, fg="bright_magenta"))
        cmd = raw.lower().strip()
        if cmd in ["exit", "quit"]:
            colorful_print("Exiting per user request.", "red")
            sys.exit(0)
        if cmd == "chat":
            knowledge_chat_session()
            continue
        if cmd in ["help", "?"]:
            colorful_print("Enter a comma-separated list of valid column names (e.g., X1, X2).", "yellow")
            continue
        cleaned = re.sub(r"\b(the|input|feature|features|is|are)\b", "", raw, flags=re.IGNORECASE)
        cols = [c.strip() for c in cleaned.split(",") if c.strip()]
        missing = [col for col in cols if col not in df.columns]
        if missing:
            colorful_print(f"ERROR: Columns {missing} not found in dataset. Try again.", "red")
            continue
        return cols

@click.command()
def run_cli():
    greet_user()
    
    # Load training CSV data or CIF file.
    df = None
    cif_data = None
    while df is None and cif_data is None:
        user_input = click.prompt(click.style("Hello Scientist, thanks for using this software. Enter path to your training CSV or CIF file, or 'help' to get around, or 'chat' to ask questions or learn about anything, 'exit' to leave :(:", fg="bright_green"))
        cmd = user_input.lower().strip()
        if cmd in ["exit", "quit"]:
            colorful_print("Exiting.", "red")
            return
        if cmd == "chat":
            knowledge_chat_session()
            continue
        if cmd in ["help", "?"]:
            colorful_print("Provide a valid CSV file path (e.g., /path/to/data.csv) or CIF file path (e.g., /path/to/structure.cif).", "yellow")
            continue
        if cmd.startswith("load "):
            user_input = user_input[5:].strip()
        
        # Try loading as CSV first
        df_try = load_csv(user_input)
        if df_try is not None:
            df = df_try
            continue
            
        # Try loading as CIF
        from .data_manager import load_cif
        cif_try = load_cif(user_input)
        if cif_try is not None:
            cif_data = cif_try
            colorful_print(f"✓ CIF file loaded: {cif_data['filename']}", "green")
            colorful_print(f"Cell parameters: a={cif_data['cell_params']['a']:.3f}, b={cif_data['cell_params']['b']:.3f}, c={cif_data['cell_params']['c']:.3f}", "white")
            colorful_print(f"Atoms found: {len(cif_data['atoms'])}", "white")
            continue
            
        colorful_print("Could not load file. Try again.", "red")
    
    if df is not None:
        colorful_print("\nPreview of training data:\n", "bright_yellow")
        colorful_print(df.head().to_string(), "white")
        
        input_features = parse_feature_input(df, "Enter comma-separated input features for training")
        target_features = parse_feature_input(df, "Enter comma-separated target features for training")
        
        X = df[input_features].values
        y = df[target_features].values
    elif cif_data is not None:
        # Handle CIF file for GULP simulation
        colorful_print(f"\nCIF file loaded: {cif_data['filename']}", "bright_yellow")
        colorful_print("Proceeding to GULP simulation setup...", "white")
        
        # Set up for GULP simulation
        simulation_type = "gulp"
        simulation_parameters = f"""
CIF filename: {cif_data['filename']}
Cell parameters: a={cif_data['cell_params']['a']:.3f}, b={cif_data['cell_params']['b']:.3f}, c={cif_data['cell_params']['c']:.3f}
Alpha={cif_data['cell_params']['alpha']:.1f}, Beta={cif_data['cell_params']['beta']:.1f}, Gamma={cif_data['cell_params']['gamma']:.1f}
Atoms: {len(cif_data['atoms'])} atoms found
Symmetry operations: {len(cif_data['symmetry_ops'])} operations
"""
        X = None
        y = None
    
    # Choose model type and optionally load an existing model.
    model_type = None
    if df is not None:
        while model_type not in ["gp", "nn"]:
            val = click.prompt(click.style("Choose model type ('gp' or 'nn') or 'chat', 'help', 'exit':", fg="bright_magenta"))
            cmd = val.lower().strip()
            if cmd in ["exit", "quit"]:
                colorful_print("Exiting.", "red")
                return
            if cmd == "chat":
                knowledge_chat_session()
                continue
            if cmd in ["help", "?"]:
                colorful_print("Type 'gp' for Gaussian Process or 'nn' for Neural Network.", "yellow")
                continue
            if cmd in ["gp", "nn"]:
                model_type = cmd
            else:
                colorful_print("Invalid choice. Please type 'gp' or 'nn'.", "red")
    else:
        # For CIF files, skip model training
        model_type = "gulp"
    
    if df is not None:
        load_option = click.prompt("Do you want to load an existing model? (yes/no)", default="no")
        if load_option.lower() in ["yes", "y"]:
            model_path = click.prompt("Enter the file path to load the model", default="")
            try:
                import pickle
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                colorful_print("Model loaded successfully.", "green")
            except Exception as e:
                colorful_print(f"Error loading model: {e}. Training a new model.", "red")
                model = train_model(model_type, X, y)
        else:
            model = train_model(model_type, X, y)
    else:
        # For CIF files, no model needed
        model = None
    
    if df is not None:
        # Load unlabeled CSV data and ask for testing feature columns.
        colorful_print("Now select an unlabeled dataset (or type 'skip' to use a default subset).", "bright_cyan")
        X_unlabeled = None
        df_unlab = None
        while X_unlabeled is None:
            raw = click.prompt(click.style("Enter path to unlabeled CSV, 'skip', 'help', 'chat', 'exit':", fg="bright_magenta"), default="")
            cmd = raw.lower().strip()
            if cmd in ["exit", "quit"]:
                colorful_print("Exiting.", "red")
                return
            if cmd == "chat":
                knowledge_chat_session()
                continue
            if cmd in ["help", "?"]:
                colorful_print("Provide a valid CSV file with the same input features, or 'skip' to use the first 5 rows.", "yellow")
                continue
            if cmd in ["skip", ""]:
                X_unlabeled = X[:5]
                df_unlab = None
                colorful_print("Using first 5 rows of training data as unlabeled set.", "yellow")
                break
            if cmd.startswith("load "):
                raw = raw[5:].strip()
            df_unlab = load_csv(raw)
            if df_unlab is None:
                colorful_print("Failed to load unlabeled CSV. Try again.", "red")
                continue
            missing = [col for col in input_features if col not in df_unlab.columns]
            if missing:
                colorful_print(f"ERROR: Columns {missing} missing in unlabeled CSV. Try again.", "red")
                continue
            X_unlabeled = df_unlab[input_features].values
        
        test_features_str = click.prompt(click.style("Enter testing feature columns for unlabeled data (comma-separated, press Enter to use training input features):", fg="bright_magenta"), default="")
        if test_features_str.strip():
            test_features = [feat.strip() for feat in test_features_str.split(",")]
            if df_unlab is not None:
                X_unlabeled = df_unlab[test_features].values
        else:
            test_features = input_features
    else:
        # For CIF files, no unlabeled data needed
        X_unlabeled = None
    
    # Simulation type.
    if df is not None:
        simulation_type = click.prompt(click.style("Enter simulation type (e.g., RASPA, LAMMPS, GROMACS, GULP, zeolite) or 'chat', 'exit':", fg="bright_magenta"))
        if simulation_type.lower() in ["exit", "quit"]:
            colorful_print("Exiting.", "red")
            return
        if simulation_type.lower() == "chat":
            knowledge_chat_session()
            simulation_type = "generic_simulation"
        # Map simulation type: if the user enters something like "GCMC raspa", use "MonteCarlo".
        if "gcm" in simulation_type.lower():
            simulation_type = "MonteCarlo"
        # Check for GULP/zeolite simulations
        if simulation_type.lower() in ["gulp", "zeolite"]:
            colorful_print("✓ GULP simulation type selected", "green")
    else:
        # For CIF files, simulation type is already set to "gulp"
        simulation_type = "gulp"
    
    if df is not None:
        base_simulation_parameters = ""  # No detailed simulation parameters are requested here.
        
        # Job system.
        job_system = click.prompt(click.style("Enter job system (UGE or Slurm) or 'chat', 'exit':", fg="bright_magenta"), default="UGE")
        if job_system.lower() in ["exit", "quit"]:
            colorful_print("Exiting.", "red")
            return
        if job_system.lower() == "chat":
            knowledge_chat_session()
            job_system = "UGE"
        
        job_params = click.prompt(click.style("Enter HPC job params (e.g., '#$ -pe smp 16' or '#SBATCH --cpus-per-task=16'). If none, press Enter:", fg="bright_magenta"), default="")
        if job_params.lower() in ["exit", "quit"]:
            colorful_print("Exiting.", "red")
            return
        if job_params.lower() == "chat":
            knowledge_chat_session()
            job_params = ""
        
        folder_prompt = click.prompt(click.style("Enter folder name to save generated scripts (default: 'responses'):", fg="bright_magenta"), default="", show_default=False)
        output_folder = folder_prompt.strip() if folder_prompt.strip() else None
        
        from .active_learning import active_learning_cycle
        colorful_print("\n=== Running Active Learning Cycle ===", "bright_green")
        uncertain_point, sim_script, job_script = active_learning_cycle(
            model=model,
            model_type=model_type,
            X_unlabeled=X_unlabeled,
            simulation_type=simulation_type,
            simulation_parameters=base_simulation_parameters,
            job_system=job_system,
            job_params=job_params,
            folder=output_folder
        )
    else:
        # For CIF files, generate GULP files directly
        folder_prompt = click.prompt(click.style("Enter folder name to save generated scripts (default: 'responses'):", fg="bright_magenta"), default="", show_default=False)
        output_folder = folder_prompt.strip() if folder_prompt.strip() else "responses"
        
        # Get the original CIF file path
        cif_file_path = user_input  # This was the CIF file path entered by user
        
        # Generate GULP input file using proper generator
        from .gulp_generator import generate_gulp_input_from_cif, generate_job_script
        gulp_file_path = generate_gulp_input_from_cif(cif_file_path, output_folder)
        
        # Generate job script
        file_name = cif_data['filename']
        job_script_path = generate_job_script(file_name, output_folder)
        
        uncertain_point = None
        sim_script = os.path.basename(gulp_file_path)
        job_script = os.path.basename(job_script_path)
    
    if df is not None:
        colorful_print("\n=== Active Learning Cycle Complete ===", "bright_yellow", bold=True)
        colorful_print(f"Chosen uncertain point: {uncertain_point}", "white")
        colorful_print(f"Simulation script saved as: {sim_script}", "white")
        colorful_print(f"Job script saved as: {job_script}", "white")
        colorful_print("\nSubmit the job externally and retrieve the new results when ready.", "yellow")
    else:
        colorful_print("\n=== GULP Files Generated ===", "bright_yellow", bold=True)
        colorful_print(f"GULP input file saved as: {sim_script}", "white")
        colorful_print(f"Job script saved as: {job_script}", "white")
        colorful_print("\nDownload both files and submit the job on your HPC cluster.", "yellow")
        colorful_print("The job will calculate the lattice energy of your zeolite structure.", "cyan")
    
    colorful_print("Thank you for using osairo! Goodbye!\n", "bright_cyan", bold=True)

def main():
    run_cli()

if __name__ == "__main__":
    main()
