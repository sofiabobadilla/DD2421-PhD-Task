# DD2421-PhD-Task
Repository for the PhD assignment of course [DD2421](https://www.kth.se/student/kurser/kurs/DD2421?l=en) at KTH.


## GOAL & Overview

This PhD project aims to apply the fundamental learning of DD2421 Machine Learning course on the PhD topic of the author. 

In this case the project will be directly related to smart contract security.

Using as dataset smartbugs-curated and the recent-published exploit for the dataset sb-heist, this experiment will compare the performance of two models train with different configurations of the same dataset for vulnerable line identification.


Mix Model (MM): 
- Dataset: smartbugs-curated
- Labels: smartbugs-curated

Pure Model (PM):
- Dataset: smartbugs-curated
- Labels:  smartbugs-curated x sb-heists (exploits)

# Methodology

Each model will aim to identify vulnerable lines of code.

MM: will train using the labeles from the smartgbugs-curated dataset.

PM: will train on lines of code that are label as vulnerable if they are reported as vulnerable in the smartbugs-curated dataset and the contract has been listed as exploitable by sb-heists.

# Evaluation

Both MM and PM will be test with each others dataset to assess the performance of the model when trained with pure True Positive vulnerabilities.


# Results

Accuracy

| Test dataset| MM| PM|
|-|-|-|
| Raw smartbugs-curated| 0.88| 0.87|
| Exploitable smartbugs-curated| 0.42| 0.84|






# Reproduction
## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/DD2421-PhD-Task.git
    ```
2. Change to the repository directory:
    ```sh
    cd DD2421-PhD-Task
    ```
3. Install the required dependencies using Poetry:
    ```sh
    poetry install
    ```



## Acknowledgements
Thanks to @vivi365 for her key advise on ML for vulnerability detection.