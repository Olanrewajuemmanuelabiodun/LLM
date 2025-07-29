from langchain_openai import ChatOpenAI
from .config import OPENAI_API_KEY

def generate_job_script(job_system, simulation_script_filename, job_params=None):
    """
    Generate an HPC job submission script for UGE or Slurm using ChatOpenAI.
    The script is simple and ready for submission.
    """
    chat = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.7)

    job_params_str = job_params if job_params else ""
    
    # Check if this is a GULP simulation
    is_gulp = simulation_script_filename.endswith('.gin')
    
    if is_gulp:
        system_message = (
            "You are a highly experienced HPC assistant. Generate a SLURM job submission script for GULP simulations. "
            "Use this exact format:\n\n"
            "#!/bin/bash\n"
            "#SBATCH --job-name=[filename].gin\n"
            "#SBATCH --output=[filename].out\n"
            "#SBATCH --error=[filename].err\n"
            "#SBATCH --time=48:00:00\n"
            "#SBATCH -N 1\n"
            "#SBATCH -n 12\n"
            "#SBATCH --mem-per-cpu=3G\n\n"
            "# Run the GULP input file\n"
            "module load intel-oneapi\n"
            "module load gulp/6.1.2\n"
            "mpirun gulp < [filename].gin\n\n"
            "Replace [filename] with the actual filename without extension."
        )
    else:
        system_message = (
            "You are a highly experienced HPC assistant. Generate a complete and simple job submission script, don't forget the cluster name with #$ -q  "
            "for UGE (qsub) that sets the job name, parallel environment, and working directory; exports the necessary "
            "environment variables; and executes the simulation using the provided input script. Do not include extraneous comments."
        )
    
    user_message = (
        f"Job system: {job_system}\n"
        f"Simulation input script: {simulation_script_filename}\n"
        f"Job parameters:\n{job_params_str}\n\n"
        "Generate the HPC job submission script in plain text."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    response = chat.invoke(messages)
    return response.content
