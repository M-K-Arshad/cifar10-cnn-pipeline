# CIFAR-10 CNN Pipeline

End-to-end ML versioning project with Git, DVC and DagsHub.

A convolutional neural network classifies CIFAR-10 images into 10 classes.
The full pipeline (prepare → preprocess → train → evaluate) is reproducible with following commands:

    dvc pull
    dvc repro
