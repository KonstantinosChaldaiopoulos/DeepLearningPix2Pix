# Pix2Pix GAN for Saliency Detection: An Investigation into Fixation Prediction.   
A deep learning project for the MSc program in Democritus utilizing a pix2pix generative adversarial network and salient maps for fixation prediction.

## Repository Structure
Here are the main files in the repository and a brief explanation of their function:

`Training-Testing-Files`: Contains files related to training and testing of the model.

`Presentation.ipynb`: A Jupyter Notebook used for as an example of how you can execute the code.

`Report.pdf`: A comprehensive report on the project.

`dataloader.py`: For loading and pre-processing data.

`imports.py`: imports necessary libraries and modules.

`main.py`: The main file where the training process is initiated.

`models.py`: Contains the model definitions for the pix2pix GAN.

`presentation.pptx`: PowerPoint presentation of the project.

`requirements.txt`: Lists the  dependencies that need to be installed.

`testing.py`: A Python script used for testing the model.

`training_function.py`: Contains the training function used in the main script.

`utils.py`: Contains utility functions used across the project​1​.

## Code execution
Before running the scripts, make sure to install the necessary Python packages listed in the requirements.txt file. You can do this by running the following command in your terminal:

``!pip install -r requirements.txt``

If you run the code in google colab simply follow the instructions in the [presentation.ipynb](https://github.com/KonstantinosChaldaiopoulos/DeepLearningPix2Pix/blob/main/Presentation.ipynb) file.

# Results

## Visual Results

![Καταγραφή](https://github.com/KonstantinosChaldaiopoulos/DeepLearningPix2Pix/assets/102811531/c927f236-4844-4110-aa8b-ea86c1a8f511)

![2](https://github.com/KonstantinosChaldaiopoulos/DeepLearningPix2Pix/assets/102811531/3d032b70-84fc-4e18-9c73-2175519a9d15)

![Καταγραφή3](https://github.com/KonstantinosChaldaiopoulos/DeepLearningPix2Pix/assets/102811531/7617c7d7-7e47-4484-938d-de200c5a085d)

## Metrics Results

The model performance was evaluated using the Normalized Scanpath Saliency (NSS) score, a standard measure in the field of visual attention. Our model achieved a promising NSS score of 1.41, which is close to the minimum human baseline of 1.54, signifying a good correlation between the predicted saliency map and the human fixation data. However, it is noteworthy that certain model architectures, such as [CEDNS](https://ieeexplore.ieee.org/document/8709735) , have achieved higher accuracies of 2.39 on the CAT2000 dataset. 


# Author

Konstantinos Chaldaiopoulos - MTN2220
