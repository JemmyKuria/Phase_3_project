# PHASE_3_PROJECT


# INTRODUCTION
This project focuses on predicting the likelihood of stroke based on patient health data. Stroke is a major health concern with significant morbidity and mortality worldwide. Early identification of high-risk individuals can enable timely interventions and improve outcomes. The goal of this analysis is to build and evaluate classification models that can accurately predict stroke occurrence using demographic and medical features.


# DATA DESCRIPTION
The dataset consists of patient records including features such as age, gender, hypertension status, heart disease history, marital status, work type, smoking status, and average glucose level. The target variable indicates whether a patient has had a stroke. The data includes both categorical and numerical variables, requiring preprocessing before modeling. The dataset is not balanced between stroke and non-stroke cases, this causes the use of SMOTE which will enable the class imbalance.


# EXPLORATORY DATA ANALYSIS
Exploratory analysis was conducted to understand feature distributions and relationships with the target variable. Visualizations such as histograms, box plots, and correlation matrices helped identify patterns and potential predictors of stroke. Initial observations revealed that factors like age, hypertension, and heart disease are strongly associated with stroke occurrence. This insight guided feature selection and engineering in the modeling phase.



# DATA PREPROCESSING
The dataset required several preprocessing steps to ensure it was ready for modeling. First, a single row with the gender labeled as “Other” was dropped, as it represented a rare category and could introduce noise into the model without adding meaningful value.

Next, one-hot encoding was applied to selected binary categorical variables: gender, Residence_type, and ever_married. Using drop_first=True helped avoid multicollinearity by dropping one category from each encoded column.

For the remaining categorical variables, ordinal or label encoding was used based on the nature of the categories:

    smoking_status was mapped to numerical values:

    never smoked = 0

    formerly smoked = 1

    smokes = 2

    Unknown = 3
This preserves the general progression of exposure while allowing the model to handle the “Unknown” category explicitly.

work_type was also label encoded, with each category assigned an integer from 0 to 4:

children, Never_worked, Govt_job, Private, and Self-employed were given distinct values to reflect different occupational backgrounds.

This combination of one-hot and label encoding allowed the model to process categorical data efficiently while retaining important information. These encoded features, along with the numerical variables, were then scaled and passed into the modeling pipeline.


# MODELLING

The modeling process began with a baseline logistic regression model due to its interpretability and simplicity. Hyperparameter tuning was performed using grid search to optimize regularization parameters and class weights. Subsequently, a decision tree model was trained to capture non-linear feature interactions and improve recall for stroke cases. Finally, a random forest ensemble was implemented, achieving the highest accuracy and balanced recall-precision tradeoff by aggregating multiple decision trees. Model performance was evaluated using accuracy, precision, recall, F1-score, and ROC-AUC, with special focus on recall to minimize missed stroke cases.

!["ROC"](ROC.png)



# RESULTS
The baseline logistic regression model achieved moderate accuracy with balanced precision and recall. Hyperparameter tuning did not substantially improve its performance. The decision tree model increased recall substantially, identifying a higher proportion of stroke cases but at some cost to precision. The random forest model outperformed all others, delivering the best accuracy (94%) and strong recall (97%) and precision (92%) for stroke prediction, indicating reliable classification and minimal false negatives.

!["Random Forest Classifier"](RandomForestReport.PNG)


# DISCUSSION
The random forest’s superior performance suggests its suitability for real-world stroke risk prediction, where accurate detection of high-risk patients is critical. While decision trees provide valuable interpretability, random forests capture complex patterns more effectively. Feature importance analysis highlights key predictors, offering insights for medical professionals. Trade-offs between model complexity and interpretability should be considered based on stakeholder needs. Future work may explore additional models or techniques to further improve prediction or interpretability.

!["Feature Importance"](FeatureImportance.png)


# CONCLUSION
This project successfully developed classification models to predict stroke risk, with the random forest model emerging as the most effective. The approach balanced interpretability and predictive accuracy, addressing the critical business need for early stroke detection. The findings can assist healthcare providers in identifying at-risk patients and informing preventive strategies. Further refinement and validation on larger datasets are recommended for deployment readiness.


