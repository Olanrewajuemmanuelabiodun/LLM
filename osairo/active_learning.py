import os
import click
from .model_manager import get_most_uncertain_point
from .simulation_scripts import interactive_generate_simulation_script
from .job_scripts import generate_job_script
from .config import DEFAULT_RESPONSES_FOLDER

def colorful_print(msg, color="cyan", bold=False):
    click.secho(msg, fg=color, bold=bold)

def save_response(filename, content, folder=None):
    """
    Save the provided content to a file inside the specified folder.
    """
    target_folder = folder or DEFAULT_RESPONSES_FOLDER
    os.makedirs(target_folder, exist_ok=True)
    filepath = os.path.join(target_folder, filename)
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Saved {filename} -> {target_folder}")

def active_learning_cycle(model, model_type, X_unlabeled, simulation_type,
                          simulation_parameters, job_system, job_params=None, folder=None):
    """
    Execute one active learning iteration:
      1. Identify the most uncertain point.
      2. Display its value and ask what it represents (e.g., pressure, temperature).
      3. Prompt for a one-line simulation description.
      4. Combine these inputs into a final simulation parameter string.
      5. Generate a complete, submission-ready simulation input script via ChatOpenAI.
      6. Allow interactive modification of that script via a chat interface.
      7. Generate an HPC job submission script.
    Returns (uncertain_point, simulation_script_filename, job_script_filename).
    """
    uncertain_point, idx, uncertainty = get_most_uncertain_point(model, X_unlabeled, model_type)
    print(f"Most uncertain point index: {idx}, uncertainty: {uncertainty}")
    print(f"Most uncertain point value: {uncertain_point}")
    
    meaning = click.prompt("What does this value represent? (e.g., pressure, temperature)", default="pressure")
    sim_desc = click.prompt("Enter a one-line description of the simulation (e.g., 'N2 in cubtc')", default="")
    
    base_sim_params = (
        f"User simulation description: {sim_desc}\n"
        f"Uncertain point value: {uncertain_point} (represents {meaning})\n"
        f"# Uncertain point index: {idx}, uncertainty: {uncertainty}\n"
    )
    final_sim_params = simulation_parameters + "\n" + base_sim_params

    sim_script = interactive_generate_simulation_script(simulation_type, final_sim_params)
    sim_script_filename = f"{simulation_type}_simulation_{idx}.input"
    save_response(sim_script_filename, sim_script, folder)

    job_script = generate_job_script(job_system, sim_script_filename, job_params)
    job_script_filename = f"{job_system}_job_{idx}.sh"
    save_response(job_script_filename, job_script, folder)

    return uncertain_point, sim_script_filename, job_script_filename

def update_training_data(df, new_data):
    """
    Optionally update the training DataFrame with new simulation results.
    """
    import pandas as pd
    new_df = pd.DataFrame([new_data])
    return pd.concat([df, new_df], ignore_index=True)
