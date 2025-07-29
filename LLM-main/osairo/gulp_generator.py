import os
from pymatgen.core.structure import Structure

def generate_gulp_input_from_cif(cif_file_path, output_directory=None):
    """
    Generate GULP input file from CIF file following the exact template.
    
    Args:
        cif_file_path: Path to the CIF file
        output_directory: Directory to save the .gin file (default: same as CIF)
    """
    # Extract the base file name (without extension) to use for output files
    file_name = os.path.splitext(os.path.basename(cif_file_path))[0]
    
    # Set output directory
    if output_directory is None:
        output_directory = os.path.dirname(cif_file_path)
    
    # Read the structure from the CIF file
    structure = Structure.from_file(cif_file_path)
    
    # Define species and potentials for the GULP input
    species = [
        'species',
        'Si core 4.0000',
        'Al core 3.0000',
        'Na core 1.0000',
        'O2 core 0.86902',
        'O2 shel -2.86902',
    ]
    
    potentials = [
        'buck',
        'O2 shel O2 shel 22764.000 0.14900 27.87900 0.0 12.0',
        'buck',
        'Si core O2 shel 1283.907 0.32052 10.66158 0.0 10.0',
        'buck',
        'Al core O2 shel 1460.300 0.29912 0.00000 0.0 10.0',
        'buck',
        'Na core O2 shel 1226.840 0.30650 0.00000 0.0 10.0',
        'buck',
        'Na core Na core 7895.400 0.17090 0.00000 0.0 10.0',
        'three',
        'Si core O2 shel O2 shel 2.09724 109.47 1.9 1.9 3.5',
        'three',
        'Al core O2 shel O2 shel 2.09724 109.47 1.9 1.9 3.5',
        'spring',
        'O2 shel 74.92',
    ]
    
    # Build the GULP input file content
    gulp_input_str = 'opti conp\n'
    gulp_input_str += 'cell\n'
    gulp_input_str += f'{structure.lattice.a} {structure.lattice.b} {structure.lattice.c} '
    gulp_input_str += f'{structure.lattice.alpha} {structure.lattice.beta} {structure.lattice.gamma}\n'
    gulp_input_str += 'frac\n'
    
    # Add fractional coordinates for core atoms (Si, Al, Na)
    for site in structure:
        if site.specie.symbol in ['Si', 'Al', 'Na']:
            frac_coords = [f'{coord:.6f}' for coord in site.frac_coords]
            gulp_input_str += f'{site.specie.symbol} core {frac_coords[0]} {frac_coords[1]} {frac_coords[2]}\n'
    
    # Add fractional coordinates for oxygen atoms (core and shell)
    for site in structure:
        if site.specie.symbol == 'O':
            frac_coords = [f'{coord:.6f}' for coord in site.frac_coords]
            gulp_input_str += f'{site.specie.symbol}2 core {frac_coords[0]} {frac_coords[1]} {frac_coords[2]}\n'
    for site in structure:
        if site.specie.symbol == 'O':
            frac_coords = [f'{coord:.6f}' for coord in site.frac_coords]
            gulp_input_str += f'{site.specie.symbol}2 shel {frac_coords[0]} {frac_coords[1]} {frac_coords[2]}\n'
    
    # Append species and potentials information
    gulp_input_str += '\n'.join(species) + '\n'
    gulp_input_str += '\n'.join(potentials) + '\n'
    
    # Specify the output dump file
    gulp_input_str += f'dump {file_name}.gout\n'
    
    # Save the GULP input file in the output directory
    gulp_input_file_path = os.path.join(output_directory, f'{file_name}.gin')
    os.makedirs(output_directory, exist_ok=True)  # Ensure directory exists
    with open(gulp_input_file_path, 'w') as fl:
        fl.write(gulp_input_str)
    
    print(f'GULP input file for {file_name} generated and saved to {gulp_input_file_path}')
    return gulp_input_file_path

def generate_job_script(file_name, output_directory):
    """
    Generate SLURM job script for GULP simulation.
    """
    job_script_content = f"""#!/bin/bash
#SBATCH --job-name={file_name}.gin
#SBATCH --output={file_name}.out
#SBATCH --error={file_name}.err
#SBATCH --time=48:00:00
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --mem-per-cpu=3G

# Run the GULP input file
module load intel-oneapi
module load gulp/6.1.2
mpirun gulp < {file_name}.gin
"""
    
    job_script_path = os.path.join(output_directory, f'{file_name}_job.sh')
    os.makedirs(output_directory, exist_ok=True)  # Ensure directory exists
    with open(job_script_path, 'w') as fl:
        fl.write(job_script_content)
    
    print(f'Job script for {file_name} generated and saved to {job_script_path}')
    return job_script_path 