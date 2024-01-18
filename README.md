
# Safety Analysis of an Exoskeleton Robot by Formal Verification

## Overview
This repository hosts the research data and analysis files for a project focused on [specific research area or goal]. It includes various datasets and Jupyter Notebooks for visualization and analysis.

## Repository Structure

### Folders
- `Dataset`: Contains datasets used in the research.
- `Experiments`: Includes different experiment categories such as BMI-Based, Baseline, Gender-Based, Individual-Specific, and Top LDA.
    - `BMI-Based`: [Description of what this contains or represents]
    - `Baseline`: [Description]
    - `Gender-Based`: [Description]
    - `Individual-Specific`: [Description]
    - `Top LDA`: [Description]

[//]: # (### Files)

[//]: # (- `Metrics with BMI.xlsx`: An Excel file containing metrics related to Body Mass Index &#40;BMI&#41;.)

[//]: # (- `README.md`: The README file for the repository.)

[//]: # (- `dataset_visualization.ipynb`: A Jupyter Notebook for visualizing the datasets.)

[//]: # (- `output.png`: An output image from one of the analyses.)

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
python verification.py --config Exoskeleton/Top_LDA/config.ini
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
