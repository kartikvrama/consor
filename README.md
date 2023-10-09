# ConSOR: A Context-Aware Semantic Object Rearrangement Framework for Partially Arranged Scenes

Full code release for the IROS 2023 paper "ConSOR: A Context-Aware Semantic
Object Rearrangement Framework for Partially Arranged Scenes". Link to paper: 
[ArXiv](https://arxiv.org/abs/2310.00371).
<p align="center">
    <img src="./images/consor.png" width="850">
</p>

## Installing Packages

THis project requires python 3.8.0. Follow these steps to install the required packages:

- Install the Semantic AI2Thor Dataset module:

    ```
    pip install git+https://github.com/maxzuo/semantic-ai2thor-dataset
    ```

- (GPU required!) Install pytorch according to your system configuration from [here](https://pytorch.org/get-started/locally/).

- Install the remaining dependencies from `requirements.txt` using pip:

    ```
    pip install -r requirements.txt
    ```

### TODO list:
1. Script to run GPT-3 baseline.
2. Colab notebook to view evaluation results.
3. Update README with installation and run instructions.
4. Colab notebook to interact with ConSOR.