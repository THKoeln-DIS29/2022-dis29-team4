# Documentation

#### Welcome to Team 4's CLI Documentation

To use the CLI, you need to launch the following command:
```
python main.py 
```
To get an overview of the commands and functions available,
you can run:
```
python main.py --help
```
To use one of the availible commands f.e. use:
```
python main.py correlations
```



----------------------------------------------------------------

# Commands
<span style="font-size:1em;">
An overview of all functionalities is provided in the following section.
</span>
----------------------------------------------------------------

## Preprocess the Trainingdata
```
python main.py pre_processing
```
### ::: main.pre_processing
For the sourcecode of the pre-processing of the training data look here:
##### ::: preprocessing.pre_pro
----------------------------------------------------------------
## Correlations Map
```
python main.py correlations
```
### ::: main.correlations
<span style="font-size:1em;">
How the heatmap might look like:<br>
</span>
<img src="img/heatmap_ex.png" width="400">
<br>

For the sourcecode of the heatmap look here:
##### ::: correlation.corel
----------------------------------------------------------------
## Make a Gridsearch
```
python main.py grid_search
```
### ::: main.grid_search
For the sourcecode of the grid search look here:
##### ::: gridsearch.grid
----------------------------------------------------------------
## Preprocess the Testdata
```
python main.py pre_processing_eval
```
### ::: main.pre_processing_eval
For the sourcecode of the pre-processing of the test data look here:
##### ::: preprocessing_eval.pre_eval
----------------------------------------------------------------
## Train the Votingclassifier
```
python main.py votingclassifier
```
<span style="font-size:1em;">
Votingclassifier setup in Sci-Kit learn:<br>
</span>
<img src="img/model.png" width="800">
<br>

### ::: main.votingclassifier
For the source code of the voting classifier look here:
##### ::: voting_classifier.classy
----------------------------------------------------------------
## Apply the model
```
python main.py applymodel
```
### ::: main.applymodel
For the source code of the apply model function look here:
##### ::: apply_model.model
----------------------------------------------------------------

