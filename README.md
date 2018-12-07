# MRI_AD_Detection
Final project for Fall 2018 Deep Learning.

### INTRODUCTION
Alzheimer’s disease (AD) is the most common type of neurodegenerative disease in elderly[1-2]. The most evident character of AD pathology is neuron loss, demonstrated as brain atrophy progressing from AD signature regions (e.g. hippocampus and amygdala) to the entire cortical region, which can be measured by magnetic resonance imaging (MRI) scan. 
Convolutional neural network has been widely used for image classification tasks with excellent performance.[3-5] A well-performed CNN image classifier is usually developed based on a huge amount of training data, which is impractical for medical image classification, due to limited resource, especially brain MRI. Transform-learning, for example, using pre-training model with fine parameter tuning is a solution that can be used to develop a task-specific neural network on a small training set. The lower level layers of a well trained CNN can capture common image features like edges and corners and can be used for feature extraction for general image classification problem.[6] However, there are few of pre-trained model available for 3D images, like brain MRI scans. One way to address this challenge is to convert a 3D image to multiple 2D slicer images and utilize a pre-trained 2D image model, like pre-trained AlexNet on ImageNet.
In this project, we used 2D convolution neural network to construct prediction model for Alzheimer’s disease from 3D-MRI image data in a relatively small sample. We applied transfer learning and explored model performance for different neural network structures, different approaches of utilizing 2D image from 3D MRI scans, and data augmentation.

### How to use our code
To run code with majority vote, random slice, or fixed slice approach, please run main_nets.py in the folder called "Random_Majority_Vote" with intented parameters. 
For example, to run fixed slice approach with Resnet18:
python3 main_nets.py --model=resnet18 --approach=fixed
We support three models: alexnet, resnet18, and densenet121. 

To run code with our autoslicer approach, please run files in AutoSlicer folder directly.
We provide files called auto_slice_ResNet.py and auto_slice_DenseNet.py, which can be run directly with default values.
Alexnet is very similar to DenseNet. One can just change the pretrained model in auto_slice_DenseNet.py from densenet121 to alexnet to train autoslicer with alexnet. 

We are still actively searching for creative ways to improve the performance of autoslicer, so we contain the code in a separate folder.

### Data

We used public brain MRI data from Alzheimer’s Disease Neuroimaging Initiative (ADNI). Data can be aquired at http://adni.loni.usc.edu/
