# Slovotvir: natural experiment in lexical evolution
## Online supplement

**Manual Version Number:** 1.0.0

<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13740276.svg)](https://doi.org/10.5281/zenodo.13740276) -->

### Short Summary:
Languages evolve under various pressures, but identifying the forces that shape word popularity remains a challenge. This study investigates the mechanisms driving translation popularity in natural language using data from Slovotvir, a Ukrainian crowdsourcing platform where users propose and rank alternative translations for borrowed words. We test two competing hypotheses: selection, where intrinsic properties of translations—such as length or prior popularity—affect user preferences, and random drift, where stochastic processes dominate. Using an agent-based model and Bayesian inference, we analyze user behavior to assess the roles of length bias, frequency bias, and drift in shaping translation choices. Our findings indicate a significant preference for shorter translations, consistent with selection for brevity, while translation popularity itself does not influence user choices. Instead, users explore words in a manner consistent with random drift, selecting them in proportion to the number of likes received by their most-liked translation. These results provide empirical support for the role of selection in lexical evolution while highlighting the influence of drift in shaping word exploration dynamics.

### Folder/File Overview:

#### `data/` Folder:
Contains input and output data related to Slovotvir:
- `words_translations.csv`: Data on words and their corresponding translations
- `votes.csv`: Data on user votes for translations

#### `figures/` Folder:
Contains the figures used both in the paper and the supplementary material.

#### `notebooks/` Folder:
Contains Jupyter notebooks which can be used to reproduce the results:

- `1. Data.ipynb`: Notebook for data preprocessing.
- `2. Generative Model.ipynb`: Notebook for the analysis of the generative model.
- `3. Inference.ipynb`: Inference of the parameters of the generative model using BayesFlow
- `4. Posterior analysis.ipynb`: Analysis of the posterior distribution of the parameters.
- `5. Corpus analysis.ipynb`: Analysis of the corpus of translations.

#### `src/` Folder:
Contains Python scripts for processing data:
- `generate_data.py`: Generating data using the generative model for parameter inference.
- `extraction.py`: Extraction of Slovotvir data from the database.
- `SlovotvirModel.py`: Generative model. 
- `helpers.py`: Helper functions for data processing and analysis.

#### Additional Files:
- **requirements.txt**: Lists the necessary Python packages to run the project.

### Instructions to Run the Software:

All the analyses were run on Python 3.11.7.

#### Workflow:

1. **Install Dependencies:**
   Install the required Python packages by running:
   ```bash
   pip install -r requirements.txt
   ```

2. **Preprocess Data:**
   - Run the `1. Data.ipynb` notebook to preprocess the data.
    - Run the `generate_data.py` script to generate data using the generative model:
    ```bash
    python src/generate_data.py
    ```

3. **Run the Analysis:**
   - Run the `2. Generative Model.ipynb` notebook to analyze the generative model.
   - Run the `3. Inference.ipynb` notebook to perform parameter inference using BayesFlow.
   - Run the `4. Posterior analysis.ipynb` notebook to analyze the posterior distribution of the parameters.
   - Run the `5. Corpus analysis.ipynb` notebook to analyze the corpus of translations.
