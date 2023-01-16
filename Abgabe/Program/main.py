import typer
from rich import print

import gridsearch
import preprocessing
import preprocessing_eval
import voting_classifier
import correlation
import apply_model


app = typer.Typer(epilog="Team 4 | Max & Max", help= "Projekt 3 command line help")



@app.command()
def applymodel():
    """This command applies the previously trained model to your churn_predictions.csv file.
    
    Args:
        1 (GUI): You'll be asked by a GUI to Pick the csv where you want to apply the model to.
        2 (GUI): You'll be asked by a GUI to pick the model you want to predict with."
    Returns:
        churn_predictions (CSV): Returns a labeled csv to output/churn_predictions.csv
    


    """
    print("Starting to apply the model")
    apply_model.model()

@app.command()
def correlations():

    """This command will print you a nice heatmap from which you can read the
    correlations between the datas features.
    
    Args:
        1 (GUI): You'll be asked by a GUI to choose the corresponding CSV from which the graph will be build. 
    
    Returns:
        correlation_map (PNG): Returns an heatmap, saved in output/correlation_map.png
    
    """
    correlation.corel()

@app.command()
def grid_search():
    """ This command will print the best parameters for the Deciscion Tree and Random forest models. \n
    
    Args:
        1 (GUI): You'll be asked by a GUI to choose the corresponding CSV from which the models will get fitted.
    
    """
    gridsearch.grid()

@app.command()
def pre_processing():
    """This command will start the preprocessing training data. \n
    The files must be named traindata_(number).tar.gz . F.e. traindata_1.tar.gz. \n
    It looks for 5 files, doesn't take less or more than that. \n
    

    Args:
        1 (GUI): You'll be asked by a GUI to choose the folder from where the zipped files can be accessed.
    

    Returns:
        features (CSV): It will save the converted files as features.csv in output/

    """

    
    preprocessing.pre_pro()

@app.command()
def pre_processing_eval():
    """This command will start the preprocessing for the test data.\n
    The files must be named testdata_(number).tar.gz . F.e. testdata_1.tar.gz .\n
    It looks for 5 files, doesn't take less or more than that.\n
    It will save the converted files as eval.csv in output/ .\n

    Args:
        1 (GUI): You'll be asked by a GUI to choose the folder from where the zipped files can be accessed.

    Returns:
        eval (CSV): It will save the converted files as eval.csv in output/
    
    """
    preprocessing_eval.pre_eval()

@app.command()
def votingclassifier():
    """This command will start the training of the voting classifier.\n
    The classifiers parameters are **HARDCODED** and can only be changed via editing the voting_classifier.py file!\n
    Once the model is build it will print a F1-score.\n
    The command will save the trained model to output/voting_clf.sav .\n
    With the saved model you'll be able to appply the model to a test dataset.\n
    ----------------------------------------------------------------
    | Classifiers | Weight |
    |---|---|
    | Decision Tree |  1 |
    | Random Forest |  1 |
    | Gaussian Process Classifier | 1  |

    ----------------------------------------------------------------
    
    *Training the model will take some time!*



    Args:
        1 (GUI): You'll be asked by a GUI to choose the features.csv file for training the model.

    Returns:
        voting_clf (Pickle / sav): The command will save the trained model to output/voting_clf.sav .
    
    """
    voting_classifier.classy()

if __name__ == '__main__':
    app()