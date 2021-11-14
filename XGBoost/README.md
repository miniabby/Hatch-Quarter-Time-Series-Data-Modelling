# Import Ratio Prediction with XGBoost 
- Data Time Range: Jan 2008 - Dec 2019


## Main Modelling Stage:
1. **Initial modelling stage**: 
    - build predictive models with only US & AUS data to find the best initial model with the best and stable performance
2. **Extending stage**: 
    - extend the best initial model with competitor data(NZ & CA) added


> In `Modelling` folder:
>> 6 Month Prediction (July - Dec 2019)
>>> - `light_version_xgboost_6m_2019pred.ipynb`: code for building best initial model (6M)
>>> - `Extending 6M Model with Competitor Data.ipynb`: code for extending the best initial model (6M)
>>> - `xgboost_archive_6m_pred.ipynb`: archived experiment process (6M)

>> 12 Month Prediction (Jan - Dec 2019)
>>> - `light_version_xgboost_12m_2019pred.ipynb`: code for building best initial model (12M)
>>> - `Extending 12M Model with Competitor Data.ipynb`: code for extending the best initial model (12M)
>>> - `xgboost_archive_12m_pred.ipynb`: archived experiment process (12M)

>> `data` folder: contains the data used by the models


## Feature Set Used by Model:


### For Initial Model (with US & AUS data)

Note: Old versions are all in archive


**Lastest Version**
> `selected_features.txt`: selected features for 6m & 12m prediction including:
>> - features selected by XGBoost importance
>> - features selected by SHAP value(above zero)
>> - features selected by SHAP + Wrapper Method (provides the best results)


### For Extended Model (US & AUS data with competitors added)
> `extended_features.txt`: features (US+AUS+CA+NZ) used for 6m & 12m prediction


