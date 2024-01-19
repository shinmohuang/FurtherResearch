
# Safety Analysis of an Exoskeleton Robot by Formal Verification

## Overview
This repository hosts the research data and analysis files for a project focused on [specific research area or goal]. It includes various datasets and Jupyter Notebooks for visualization and analysis.

## Repository Structure

### Folders
- `Dataset`: Contains datasets used in the research.

[//]: # (- `Experiments`: Includes different experiment categories such as BMI-Based, Baseline, Gender-Based, Individual-Specific, and Top LDA.)

[//]: # (    - `BMI-Based`: [Description of what this contains or represents])

[//]: # (    - `Baseline`: [Description])

[//]: # (    - `Gender-Based`: [Description])

[//]: # (    - `Individual-Specific`: [Description])

[//]: # (    - `Top LDA`: [Description])
- `Experiments/Exoskeleton`: Contains the Exoskeleton model and its reduced model.
    - `Baseline`: [Description]
    - `FeatureReduced`: [Description]
- `Experiments/STS`: Contains the STS model.
    - `Baseline`: [Description]
    - `FeatureReduced`: [Description]
  - `verification.py`: The main script for running the verification experiments.

- `Results`: Contains the results of the experiments.
    - `Exoskeleton`: Contains the results of the Exoskeleton model.
        - `Baseline`: [Description]
        - `FeatureReduced`: [Description]
    - `STS`: Contains the results of the STS model.
        - `Baseline`: [Description]
        - `FeatureReduced`: [Description]



## Dependencies
```bash
pip install -r requirements.txt
```

## Installation
To run the notebooks and scripts, clone the repository and install the required dependencies.

```bash
git clone https://github.com/shinmohuang/FurtherResearch.git
```

## Usage
To run the verification for the Baseline experiment (for example), run the following command from the root directory of the repository.

### Run with Default (Exoskeleton Baseline)

```bash
cd FurtherResearch/Experiments
python verification.py
```

### Run Reduced Model (Exoskeleton Reduced)

```bash
cd FurtherResearch/Experiments
python verification.py --config Exoskeleton/FeatureReduced/config.ini
```

### Run STS Model

```bash
cd FurtherResearch/Experiments
python verification.py --config STS/Baseline/config.ini
```

## Contributing
Contributions to this project are welcome. Please submit pull requests for any enhancements.

## License
Specify the license under which this project is released.

## Contact
For any queries regarding this project, please contact [Your Contact Information].
