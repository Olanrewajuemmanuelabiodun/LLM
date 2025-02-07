# osairo/simulation_scripts.py
from langchain_openai import ChatOpenAI
from .config import OPENAI_API_KEY
import click

def generate_simulation_script(simulation_type, simulation_parameters):
    """
    Generate a submission-ready simulation input script using ChatOpenAI.
    The script is formatted for submission (e.g., a clean RASPA GCMC script with no Box section if a MOF is used).
    """
    chat = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.0)

    # Extensive system message covering common simulation types.
    if simulation_type.lower() in ["raspa", "gcmc", "montecarlo"]:
        system_message = (
            "You are a world-class molecular simulation expert with deep knowledge of RASPA. "
            "Generate a complete, submission-ready RASPA simulation input script for a Grand Canonical Monte Carlo (GCMC) simulation. "
            "Please adhere to the following guidelines:\n\n"
            "1. SimulationType: Set this to 'MonteCarlo'.\n"
            "2. NumberOfCycles: Specify the total number of simulation cycles (e.g., 200000).\n"
            "3. NumberOfInitializationCycles: Set the number of initialization cycles (if the ensemble is μVT, ensure this is at least 1000).\n"
            "4. PrintEvery: Define how frequently output is printed (e.g., 1000).\n"
            "5. ContinueAfterCrash: Should be set to 'no'.\n"
            "6. WriteBinaryRestartFileEvery: Specify a value (e.g., 5000).\n"
            "7. Forcefield: Use an appropriate force field (e.g., 'UFF').\n"
            "8. RemoveAtomNumberCodeFromLabel: Set to 'yes'.\n"
            "9. Framework: Under this section, provide the FrameworkName (e.g., 'Cu-BTC'), UnitCells (e.g., '1 1 1'), ExternalTemperature (e.g., 298), and ExternalPressure. "
            "   If an uncertain point (representing pressure) is provided, use that value for ExternalPressure.\n"
            "10. Component: For each component, specify MoleculeName (e.g., 'N2'), MoleculeDefinition (e.g., 'TraPPE'), TranslationProbability, ReinsertionProbability, "
            "    RotationProbability, SwapProbability, and CreateNumberOfMolecules (should be 0 for GCMC simulations).\n"
            "11. Do not include any Box section if the simulation involves a MOF.\n\n"
            "Output the script in plain text with no extra commentary, units, or headings."
        )
    elif simulation_type.lower() in ["lammps"]:
        system_message = (
            "You are an expert in molecular dynamics using LAMMPS. Generate a complete, submission-ready LAMMPS input script "
            "with the following guidelines:\n\n"
            "1. Specify the simulation units (e.g., 'real', 'metal', or 'lj').\n"
            "2. Define the simulation box and boundary conditions.\n"
            "3. Set up the atom style and create the atoms using appropriate lattice parameters.\n"
            "4. Define the interatomic potential (pair_style and pair_coeff) for the system.\n"
            "5. Initialize velocities and define the time step and number of simulation steps.\n"
            "6. Include output commands to record data, such as thermo and dump commands.\n\n"
            "Output the script in plain text with no extra commentary."
        )
    elif simulation_type.lower() in ["gromacs", "gromacs md"]:
        system_message = (
            "You are an expert in molecular dynamics using GROMACS. Generate a complete, submission-ready GROMACS input script "
            "that includes all necessary configuration parameters:\n\n"
            "1. Specify the topology file, and create an .mdp file with parameters for temperature coupling, pressure coupling, "
            "time step, simulation length, and output frequencies.\n"
            "2. Define the simulation box and, if necessary, the solvent model.\n"
            "3. Include settings for energy minimization, equilibration, and production runs.\n\n"
            "Output the script in plain text with no extra commentary."
        )
    else:
        system_message = (
            "You are an expert in molecular simulations. Generate a complete, submission-ready simulation input script "
            "that includes all necessary parameters for a typical simulation. Output the script in plain text with no extra commentary."
        )

    # User message incorporates the simulation parameters provided.
    user_message = (
        f"Simulation type: {simulation_type}\n"
        f"Parameters:\n{simulation_parameters}\n\n"
        "Generate the simulation input script."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    response = chat.invoke(messages)
    return response.content

def interactive_generate_simulation_script(simulation_type, simulation_parameters):
    """
    Generate an initial sample simulation input script and allow interactive modification.
    First, prompt for the simulation ensemble (e.g., 'NVT', 'NPT', 'μVT'). If the ensemble is μVT, automatically add parameters so that:
      - SimulationType becomes "MonteCarlo"
      - NumberOfInitializationCycles is set to 1000.
    Then generate the sample script, display it (without extra headings), and allow iterative modifications via a chat interface.
    Return the final script.
    """
    ensemble = click.prompt("Enter ensemble type for simulation (e.g., 'NVT', 'NPT', 'μVT')", default="NVT")
    if ensemble.lower() in ["μvt", "muvt", "mu vt"]:
        simulation_parameters += "\nSimulationType MonteCarlo\nNumberOfInitializationCycles 1000\n"
    else:
        simulation_parameters += f"\nEnsemble: {ensemble}\n"
    
    click.echo("\nGenerating initial sample simulation input script...\n")
    current_script = generate_simulation_script(simulation_type, simulation_parameters)
    click.echo(current_script)
    click.echo("")
    
    def prompt_yes_no(message, default="no"):
        while True:
            ans = click.prompt(message, default=default).strip().lower()
            if ans in ["yes", "y"]:
                return True
            elif ans in ["no", "n"]:
                return False
            else:
                click.echo("Please enter 'yes' or 'no'.")
    
    if prompt_yes_no("Would you like to modify the generated script? (yes/no)", default="no"):
        chat = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.0)
        while True:
            mod_text = click.prompt("Enter your modifications (or type 'done' to finish):", default="")
            if mod_text.lower() == "done" or not mod_text.strip():
                break

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that helps update simulation input scripts. Modify the current script based on the user's requested changes. "
                        "Return a submission-ready script in plain text without extra commentary."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Current script:\n{current_script}\n"
                        f"Requested modifications:\n{mod_text}\n"
                        "Provide the updated simulation input script."
                    )
                }
            ]
            response = chat.invoke(messages)
            current_script = response.content
            click.echo("")
            if not prompt_yes_no("Would you like to modify the script further? (yes/no)", default="no"):
                break
    
    return current_script
