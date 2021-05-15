# Round Cell Tumor Identifier.

## Introduction
The aims of this project were threefold:
- Create a novel veterinary round cell tumor classification dataset.
- Develop and train models to classify round cell tumors on this dataset.
- Productionize a trained model using FastAPI and Streamlit in the form of an app.

## Milestones Completed
- Novel veterinary round cell tumor classification dataset collated and labeled using LabelStudio. Contains 6038 images.
- Dataset stored and tracked using AWS S3 and DVC.
- A baseline CNN model trained and validated, returning a validation accuracy of 86%. 
- Hyperparameter sweep of the CNN model conducted using WandB.

## Milestones in Progress

- Experiment with a pretrained and fine-tuned Resnet model as the classifier.
- Move training onto AWS using and EC2 instance, and implement multi-GPU training.

## Milestones Todo
- Productionize trained model.
- Peer review labels.
- Incorporate contrastive learning, learning under privileged information into model training.
- Generate test set accuracy for models and compare.

### Notes
- The dataset will be published publicly, however I am waiting on verification of the labels from a second and third pathologist.
- The code in this repo is very rough, and I would say in draft form. My plan is clean up this code, build a robust CI/CD pipeline, improve the documentation.
- Much of the work on this project is rough, a draft version, and needs cleaning up for public utilization. 