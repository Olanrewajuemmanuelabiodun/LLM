import pandas as pd
import re

def load_csv(filepath: str):
    """
    Load a CSV file and return a DataFrame.
    If loading fails, print an error and return None.
    """
    try:
        df = pd.read_csv(filepath)
        print("CSV loaded successfully.")
        print("Columns:", df.columns.tolist())
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def load_cif(filepath: str):
    """
    Load a CIF file and extract cell parameters and atomic coordinates.
    """
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract cell parameters
        cell_params = {}
        cell_patterns = {
            'a': r'_cell_length_a\s+([\d.]+)',
            'b': r'_cell_length_b\s+([\d.]+)',
            'c': r'_cell_length_c\s+([\d.]+)',
            'alpha': r'_cell_angle_alpha\s+([\d.]+)',
            'beta': r'_cell_angle_beta\s+([\d.]+)',
            'gamma': r'_cell_angle_gamma\s+([\d.]+)'
        }
        
        for param, pattern in cell_patterns.items():
            match = re.search(pattern, content)
            if match:
                cell_params[param] = float(match.group(1))
        
        # Extract atomic coordinates
        atoms = []
        coord_section = re.search(r'loop_\s*_atom_site_label.*?(?=\n\n|\Z)', content, re.DOTALL)
        if coord_section:
            lines = coord_section.group(0).split('\n')
            for line in lines:
                if line.strip() and not line.startswith('_') and not line.startswith('loop_'):
                    parts = line.split()
                    if len(parts) >= 4:
                        atoms.append({
                            'label': parts[0],
                            'element': parts[1],
                            'x': float(parts[2]),
                            'y': float(parts[3]),
                            'z': float(parts[4])
                        })
        
        # Extract symmetry operations
        symmetry_ops = []
        sym_section = re.search(r'loop_\s*_symmetry_equiv_pos_as_xyz.*?(?=\n\n|\Z)', content, re.DOTALL)
        if sym_section:
            lines = sym_section.group(0).split('\n')
            for line in lines:
                if line.strip() and not line.startswith('_') and not line.startswith('loop_'):
                    symmetry_ops.append(line.strip().strip("'"))
        
        return {
            'cell_params': cell_params,
            'atoms': atoms,
            'symmetry_ops': symmetry_ops,
            'filename': filepath.split('/')[-1].replace('.cif', '')
        }
    except Exception as e:
        print(f"Error loading CIF: {e}")
        return None
