# Analyzing-Skin-Tone-Bias-in-Deep-Neural-Networks-for-Skin-Condition-Diagnosis
Analyzing Skin Tone Bias in Deep Neural Networks for Skin Condition Diagnosis
1. Fitzpatrick17k.csv - Original Dataset
2. data_Visualization.ipynb - data distributions in orinigal Dataset
3. Preprocessing.ipynb - Preprocessing of orinigal Dataset
4. data_cleaning.ipynb - data cleaning of the preprocessed data
5. final_dataset.ipynb - final dataset we made after all preprocessing and datacleaning. we will use this for classification
6. DNNmodelcode.zip - has .py files of VGG16, DenseNet and InceptionNet models. The classification models are defined on final_dataset. 
7. Data_Augmentation.ipynb - data augmentation to make 3partition labels equal and save a new dataset, 3partitionbalanced_dataset.csv 
8. 3partitionbalanced_dataset.ipynb - updated dataset after 3partitions are balanced. use VGG16 from DNNmodelcode just change dataset from finaldataset to this dataset
9. 50imageseach_VGG16.ipynb - data augmentation to make 114 label equal save as a new dataset, equalized_dataset.csv also has VGG16 model in it
10. contrastchnage.ipynb - changed contarst and VGG16 model on finaldataset.csv
