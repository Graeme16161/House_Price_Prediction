# Housing Price Prediction 

Project to predict housing prices with kaggle data and machine learning techniques

## Important File List

 * data_description.txt - descriptions for data features from kaggle
 * Housing_Data_EDA.rmd - R markdown file for initial data EDA
 * Keleher_model_compare.ipynb – Reads in all datasets and fits a Lasso and XGBoost model for each with 10 fold cross validation. The datasets were generated in Clean_code_Datasets.ipynb
 * Clean_code_Datasets.ipynb – Generates different datasets from Kaggle data using various feature engineer techniques
 * clustering_categoric_features.ipynb – Performs Agglomerative clustering on different levels of categoric features and subsequently reduces dimensionality by merging levels 
 * Model_Validation_final.ipynb – validates the models found in Keleher_model_compare.ipynb
