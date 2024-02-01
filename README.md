# Machine learning based prediction of long-term energy consumption and overheating under climate change impacts using urban building energy modeling

This is the official implementation of the our paper:

Ilkim Canli, Eren Gökberk Halacli, Sevval Ucar, Orcun Koral Iseri, Feyza Yavuz, Dilara Güney, Ayca Duran, Cagla Meral Akgul, Sinan Kalkan, Ipek Gursel Dino,
"Machine learning based prediction of long-term energy consumption and overheating under climate change impacts using urban building energy modeling",
under review, 2024.

## A Brief Overview of The Paper

In our paper, we propose an ML-based, UBEM-assisted approach to make precise heating energy use and indoor overheating predictions for years 2020, 2050 and 2080. We trained multi-layer perceptrons (MLPs) using all three year's data and obtained 0.98 and 0.96 R^2 scores for heating energy use and indoor overheating respectively.

## Dependencies
- WandB
- numpy
- python3
- pandas
- matplotlib.pyplot
- pytorch
- sklearn
- yaml

## Installation and Use

Follow these steps to set up and install the project:

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/your-project.git
    ```

2. Install the dependencies using `pip` and the provided `requirements.txt` file in energy-journal directory:

    ```bash
    pip install -r requirements.txt
    ```

3. Fill the blanks in `config.ini` in codes directory.

4. Run the project in codes directory:

    ```bash
    python3 codes.py
    ```


## Experimental Setups

Our results indicated on paper are obtained with the following configurations:

|Prediction| batch_size |layer_num | layer_size | learning_rate |
| --- | --- | --- | --- | --- |
| Heating End Use | 32 | 6 | 48 | 0.0004869 |
| Indoor Overheating | 64 | 5 | 64 | 0.003542 |

# Full Readme is Being Prepared
