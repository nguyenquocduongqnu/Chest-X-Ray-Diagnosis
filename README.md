# Chest-X-Ray-Diagnosis: VinBigData Chest X-ray Abnormalities Detection

## Installation
```!pip install -r requirements.txt```

## Training model

### Pre-Trained Model
We can download the weight file of the pre-trained YOLOv5 model [here](https://drive.google.com/drive/u/2/folders/1QdM5d4I33AhSAIkcMASns4DEIgHTHJjS). Then move the **best.pt** file to the models folder.

### OR Train Model
We can train the YOLOv5 model by re-training the **Train_YOLOv5_ColabPro.ipynb** file in the **Train_Model** folder to get the **best.pt** weight file. We use google colab pro with high-RAM runtime and GPU select mode (VGA NVIDIA Tesla P100 16GB). The process of training the model took 3 hours and 20 minutes.

## Run Demo
 ```python .\app.py```
 
### Result

 ![All text](templates/demo.PNG)
 
 
 ***Updating ...***
