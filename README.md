# Weak Supervision for Text Classification: Replicating IR-TF-IDF and Word2Vec
This repository contains the implementation for replicating the results of text classification using two different methods: IR-TF-IDF and Word2Vec. The project aims to evaluate the performance of these methods on two datasets: The New York Times (NYT) and the 20 Newsgroup dataset, with both coarse-grain and fine-grain labels.

## Getting Started
To replicate the results, follow these steps:

# Prerequisites
Ensure you have Anacodna installed on your system to manage the project dependencies.

# Cloning the Repository
Clone this repository (which contains all the necessary data) to your local machine using:
```
git clone https://github.com/eugenekim3107/dsc180a-project.git
```

# Setting Up the Conda Environment
Create a new Conda environment using the `requirements.txt` file:
```
conda create --name <env-name> --file requirements.txt
```
Activate the environment:
```
conda activate <env-name>
```
# Running the Scripts
You can replicate the results for each method by running the corresponding Python script.

**IR-TF-IDF Method**
To run the IR-TF-IDF method, use:
```
python3 IRTFIDF.py
```

**Word2Vec Method**
To run the Word2Vec method, use:
```
python3 Word2Vec.py
```

# Results
Each script will output eight different scores for the respective dataset:
- Micro-F1 Score
- Macro-F1 Score

These scores are provided for both the NYT and the 20 Newsgroup datasets, under coarse-grain and fine-grain labels.