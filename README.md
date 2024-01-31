# Machine learning based prediction of long-term energy consumption and overheating under climate change impacts using urban building energy modeling

This is the official implementation of the our paper:

Ilkim Canli, Eren Gökberk Halacli, Sevval Ucar, Orcun Koral Iseri, Feyza Yavuz, Dilara Güney, Ayca Duran, Cagla Meral Akgul, Sinan Kalkan, Ipek Gursel Dino,
"Machine learning based prediction of long-term energy consumption and overheating under climate change impacts using urban building energy modeling",
under review, 2024.

# A Brief Overview of The Paper

In our paper, we propose an ML-based, UBEM-assisted approach to make precise heating energy use and indoor overheating predictions for years 2020, 2050 and 2080. We trained multi-layer perceptrons (MLPs) using all three year's data and obtained 0.98 and 0.96 R^2 scores for heating energy use and indoor overheating respectively.

# Dependencies
- WandB
- numpy
- python3
- pandas
- matplotlib.pyplot
- pytorch
- sklearn
- yaml

# Using codes.py

- Append your WandB api key to 
```
wandb.login(key = "")
```
in codes.py

- Specify the desired task:
    - If you want to predict heating end use: "heat"
    - If you want to predict indoor overheating: "iod"

- Specify the path you want to save your trained model in 
```
if (task=="heat"):
        torch.save(net.state_dict(),f"") # specify path to save your heat model
    elif (task=="iod"):
        torch.save(net.state_dict(),f"") # specify path to save your iod model
```
at the end of codes.py

# Heating End Use Experimental Setup

Our results indicated on paper are obtained with the following configurations:

| batch_size |layer_num | layer_size | learning_rate |
| --- | --- | --- | --- |
| 32 | 6 | 48 | 0.0004869 |

# Indoor Overheating Experimental Setup

Our results indicated on paper are obtained with the following configurations:

| batch_size |layer_num | layer_size | learning_rate |
| --- | --- | --- | --- |
| 64 | 5 | 64 | 0.003542 |

# Full Readme is Being Prepared
