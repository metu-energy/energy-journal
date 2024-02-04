# Machine learning based prediction of long-term energy consumption and overheating under climate change impacts using urban building energy modeling

This is the official implementation of our paper:

Ilkim Canli, Eren Gökberk Halacli, Sevval Ucar, Orcun Koral Iseri, Feyza Yavuz, Dilara Güney, Ayca Duran, Cagla Meral Akgul, Sinan Kalkan, Ipek Gursel Dino,
"Machine learning based prediction of long-term energy consumption and overheating under climate change impacts using urban building energy modeling",
under review, 2024.

## A Brief Overview of The Paper

In our paper, we propose an ML-based, UBEM-assisted approach to make precise heating energy use and indoor overheating predictions for years 2020, 2050 and 2080. We trained multi-layer perceptrons (MLPs) using all three year's data and obtained 0.98 and 0.96 $R^2$ scores for heating energy use and indoor overheating respectively.


## How to Use the Code & the Dataset

### Dependencies
- WandB
- numpy
- python3
- pandas
- matplotlib.pyplot
- pytorch
- sklearn
- yaml

### Installation and Use

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

### Experimental Setups

Our results indicated on paper are obtained with the following configurations:

|Prediction| batch_size |layer_num | layer_size | learning_rate |
| --- | --- | --- | --- | --- |
| Heating End Use | 32 | 6 | 48 | 0.0004869 |
| Indoor Overheating | 64 | 5 | 64 | 0.003542 |

## How to Cite

Ilkim Canli, Eren Gökberk Halacli, Sevval Ucar, Orcun Koral Iseri, Feyza Yavuz, Dilara Güney, Ayca Duran, Cagla Meral Akgul, Sinan Kalkan, Ipek Gursel Dino,
"Machine learning based prediction of long-term energy consumption and overheating under climate change impacts using urban building energy modeling",
under review, 2024.

```
@article{ML_UBEM2024, 
  title={Machine learning based prediction of long-term energy consumption and overheating under climate change impacts using urban building energy modeling},
  author={Ilkim Canli, Eren Gökberk Halacli, Sevval Ucar, Orcun Koral Iseri, Feyza Yavuz, Dilara Güney, Ayca Duran, Cagla Meral Akgul, Sinan Kalkan, Ipek Gursel Dino}, 
  journal={under review}, 
  year={2024}, 
}
```

## License
This project is released under the [Apache 2.0 license](LICENSE). Please see also the licences of each API provided under each directory.

## Contact

This repo is maintained by the [METU Energy Research Group](http://metu-energy.github.io).
