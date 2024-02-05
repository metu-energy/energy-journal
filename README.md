# Machine learning based prediction of long-term energy consumption and overheating under climate change impacts using urban building energy modeling

This is the official implementation of our paper:

Ilkim Canli, Eren Gökberk Halacli, Sevval Ucar, Orcun Koral Iseri, Feyza Yavuz, Dilara Güney, Ayca Duran, Cagla Meral Akgul, Sinan Kalkan, Ipek Gursel Dino,
"Machine learning based prediction of long-term energy consumption and overheating under climate change impacts using urban building energy modeling",
under review, 2024.

## A Brief Overview of The Paper

In our paper, we propose an ML-based, UBEM-assisted approach to make precise heating energy use and indoor overheating predictions for years 2020, 2050 and 2080. We trained multi-layer perceptrons (MLPs) using all three year's data and obtained 0.98 and 0.96 $R^2$ scores for heating energy use and indoor overheating respectively. Our datasets used in the paper are available on this GitHub page in the datasets folder. Datasets contain descriptive parameters (Year, Version, unit_id), output parameters (heating energy use as "Q_heating" and indoor overheating as "IOD") and input parameters (rest). Descriptive parameters are not used in training process; however, they are included to identify zones.

## How to Use the Code & the Dataset
There are three datasets for years 2020, 2050 and 2080, presented in csv format. Libraries and functions (_data_prep_, _set_scalers_ and _dataset_gen_) required to process the datasets are present in /codes/codes.py. Classes for model and loss (_Model_ and _RMSELoss_ classes) and a function to plot scatter plots (_plotter_) are also included in there. Our trained models with the configurations indicated in Experimental Setups section can be found in the models directory. _heat_model_ and _iod_model_ predicts heating end use and indoor overheating respectively.

A more detailed explanation is provided in the following sections if you want to develop models from scratch using our codes.

### Dependencies
- WandB
- numpy
- python3
- pandas
- matplotlib.pyplot
- pytorch
- sklearn
- yaml

### Installation & Use

Follow these steps to set up and install the project:

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/your-project.git
    ```

2. Install the dependencies using `pip` and the provided `requirements.txt` file in energy-journal directory:

    ```bash
    pip install -r requirements.txt
    ```

3. Fill in the blanks in `config.ini` in codes directory.

   * __wandb_api_key__ : Append your WandB api key to track train and test processes.
   * __task__ : If you want to predict heating energy use, write _heat_. If you want to predict indoor overheating, write _iod_.
   * __path_and_name_to_model__ : Specify the path you want to save your model to along with the name of the model. Extension of the model should be _.pt_.
   * __project_name__ : This is the name of the project when you are creating a new run with Wandb.
   * __entity__ : This is the name of a team or a username when you are creating a new run with WandB.
   * __number_of_experiments_to_run__ : Number of experiments to run with WandB.

4. (Optional) Specify new intervals for hyperparameters in /codes/config.ini.

5. Run the project in codes directory:

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
