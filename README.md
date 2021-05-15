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

