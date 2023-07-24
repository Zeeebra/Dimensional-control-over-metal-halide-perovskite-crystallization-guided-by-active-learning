# Dimensional control over metal halide perovskite crystallization guided by active learning
The project was performed in Lawrence Berkeley National Lab

The data and results were shown and discussed in Manuscript.pdf and Supporting information.pdf.

Data folder contains the whole datasets (cleaned and combined) and the records of all individual ruans of this active learning experiment.

Python folder - 3D primary screening: see "Primary Screening of 3D Reaction-Composition 203 Space" section the manuscrip. The code is to visualize the outcomes of the perovskite crystallization, to identify the scope the high dimention (6D) search of the experiment space.

Python folder 6D AL: see "Modified Workflow To Screen Additives in Six-
262 Dimensional Reaction-Composition Space". This section is exploring morpholinium perovskite crystallization by using additive etc, with 6 experiment parameters (6D) using active learning. All machine learning code used in the project is saved in this folder. E.g., "Generation of the 6D reaction pool.ipynb" is used to generate the active learning pool; "Active Learning_4th_(6D)_Random Forrest.ipynb" is used to perform the 4th active learning iteration; Other jupyter notebooks contained in this folder are used for visualizing the experiemnt outcomes, calculating CV accuracies of the model, calculating AL metrics, checking the upper and lower predicting accuracy limit of the dataset, model interpretation (checking feature importance to draw scientifc conclusion), etc. The detail methods for algorithms used in those notebooks are discussed in the Supporting information.pdf.
