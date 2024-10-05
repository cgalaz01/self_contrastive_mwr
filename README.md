# Multi-Tiered Self-Contrastive Learning for Medical Microwave Radiometry (MWR) Breast Cancer Detection

## Setup
To set up the project, follow these steps:

1. Install Anaconda from the official website: [Anaconda](https://www.anaconda.com/products/individual).

2. Clone the repository to your local machine:
    ```
    git clone https://github.com/cgalaz01/self_contrastive_mwr.git
    ```

3. Navigate to the project directory:
    ```
    cd self_contrastive_mwr
    ```

4. Create a new conda environment using the provided `environment.yml` file:
    ```
    conda env create -f environment.yml
    ```

5. Activate the conda environment:
    ```
    conda activate self_contrastive_mwr
    ```


## Model Training
To train and evaluate a model run the Python script 'run_trianing.py':

1. Navigate to the project's source code.
    ```
    cd src
    ```

2. Run the Python script with the desired command-line arguments. For example, to run the script with the default values for `model_type` and `contrastive_type`, use the following command:
    ```
    python run_training.py
    ```

3. If you want to specify different values for the arguments, use the `--model_type` (either 'base', 'local', 'regional', 'global' or 'joint') and `--contrastive_type` (either 'none', 'contrastive', 'triplethard', 'tripletsemihard' or 'npairs') flags followed by the desired values. _Note: 'joint' model expects the respective 'local', 'regional' and 'global' models to be trained first._ For example:
    ```
    python run_training.py --model_type local --contrastive_type none
    ```

## Contributing
Contributions are welcome! Here's how you can contribute to the project:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature-name`.
3. Make your changes and commit them: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature/your-feature-name`.
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
