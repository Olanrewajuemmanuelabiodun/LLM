from langchain_openai import ChatOpenAI
from .config import OPENAI_API_KEY

def generate_job_script(job_system, simulation_script_filename, job_params=None):
    """
    Generate an HPC job submission script for UGE or Slurm using ChatOpenAI.
    The script is simple and ready for submission.
    """
    chat = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.7)

    job_params_str = job_params if job_params else ""
    
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
