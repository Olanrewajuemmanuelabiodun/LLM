from setuptools import setup, find_packages

setup(
    name='osairo',
    version='0.1.0',
    description='All-purpose molecular modeling, quantum chemistry, ML active learning, and HPC job script generator',
    author='Etinosa Osaro',
    author_email='eosaro@nd.edu',
    packages=find_packages(),
    install_requires=[
        'click',
        'pandas',
        'numpy',
        'scikit-learn',
        'langchain-openai',
        'gpflow',
        'tensorflow'
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'osairo=osairo.cli:run_cli'
        ],
    },
)
