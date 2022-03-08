# 2022Spring-Project2-STAT5242
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1g6XZovBwKhdrb3UX2Ec9m4v3jBrAit9o?usp=sharing)

# Climate Prediction Challenges
        
## **Physics-Guided Deep Learning Lake Thermal Stratification Modelling**

<span style="color:red">Mohammed Aqid Khatkhatay, Tianxiao Shen, Ziwen Ye  (Group 6)
     
The goal of this project is to perform Lake Thermal Stratification Modelling and to explore the physics-guided deep learning models proposed in Read et al (2019) and Jia et al (2019).
     
 

Source: 
## Code Run Instructions
Install the necessary requirements by running the following command.

~~~python

!pip install sciencebasepy
!mkdir models
!mkdir models/Data
!mkdir Output
!wget https://raw.githubusercontent.com/aqid98/ClimatePredictionChallengesProject2/main/codes/dataset.py
!wget https://raw.githubusercontent.com/aqid98/ClimatePredictionChallengesProject2/main/codes/loss.py
!wget https://raw.githubusercontent.com/aqid98/ClimatePredictionChallengesProject2/main/codes/trainer.py
!wget https://raw.githubusercontent.com/aqid98/ClimatePredictionChallengesProject2/main/codes/utils.py
!wget https://raw.githubusercontent.com/aqid98/ClimatePredictionChallengesProject2/main/architectures/LSTM.py -P "/content/models"
~~~

next run the code to upload the dataset.  
~~~python
%%capture

~~~

    
## Organisation of this directory 

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
├── images/        
└── output/
        
```
    
## Data Description
        
Climate change has been shown to influence lake temperatures in different ways. To better understand the diversity of lake responses to climate change and give managers tools to manage individual lakes, we focused on improving prediction accuracy for daily water temperature profiles in 68 lakes in Minnesota and Wisconsin during 1980-2018.
The data are organized into these items:

+ Spatial data - One shapefile of polygons for all 68 lakes in this study (.shp, .shx, .dbf, and .prj files)
+ Model configurations - Model parameters and metadata used to configure models (1 JSON file, with metadata for each of 68 lakes, indexed by "site_id")
+ Model inputs - Data formatted as model inputs for predicting temperature
> a. Lake Mendota model inputs - Tables with 1 row per timestep for weather data and ice flags (2 comma-delimited files)
> b. Sparkling Lake model inputs - Tables with 1 row per timestep for weather data and ice flags (2 comma-delimited files)
> c. Historical model inputs for 68 lakes - Tables with 1 row per timestep for weather data and ice flags, with two files for each lake (138 comma-delimited files, compressed into 2 zip files)
+ Training data - Data used to train or calibrate predictive models
> a. Lake Mendota training data - Tables with 1 row per date and depth, with the corresponding observed water temperature (3 comma-delimited files)
> b. Sparkling Lake training data - Tables with 1 row per date and depth, with the corresponding observed water temperature (3 comma-delimited files)
> c. Historical training data for 68 lakes - Tables with 1 row per date, depth, and site_id, with the corresponding observed water temperature (1 comma-delimited file)
+ Prediction data - Predictions from PGDL, DL, and PB models
> a. Lake Mendota predictions - Tables with 1 row per date, a column for predicted temperature at each depth, and experiment metadata (10 comma-delimited files)
> b. Sparkling Lake predictions - Tables with 1 row per date, a column for predicted temperature at each depth, and experiment metadata (10 comma-delimited files)
> c. Historical predictions for 68 lakes - Tables with 1 row per date and depth, with the corresponding observed water temperature (4 comma-delimited files for each lake compressed into 68 zip files)
+ Model evaluation - test data and overall assessment of model performance
> a. Lake Mendota evaluation - Tables with 1 row per date, a column for predicted temperature at each depth, and experiment metadata (3 comma-delimited files)
> b. Sparkling Lake evaluation - Tables with 1 row per date, a column for predicted temperature at each depth, and experiment metadata (2 comma-delimited files)
> c. Historical evaluation for 68 lakes - Tables with 1 row per date and depth, with the corresponding observed water temperature (4 comma-delimited files for each lake compressed into 68 zip files)
        


    
    
## References 

Read, J.S., Jia, X., Willard, J., Appling, A.P., Zwart, J.A., Oliver, S.K., Karpatne, A., Hansen, G.J.A., Hanson, P.C., Watkins, W., Steinbach, M., and Kumar, V., 2019, Data release: Process-guided deep learning predictions of lake water temperature: U.S. Geological Survey data release, https://doi.org/10.5066/P9AQPIVD.
