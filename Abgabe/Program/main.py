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
    """This command applies the previously trained model to your eval.csv file.
    
    Args:
        1.You'll be asked by a GUI to Pick the csv where you want to apply the model to.
        2.You'll be asked by a GUI to pick the model you want to predict with."


    """
    print("Starting to apply the model")
    apply_model.model()

@app.command()
def correlations():

    """This command will print you a nice heatmap from which you can read the
    correlations between the datas features.
    
    Args:
        You'll be asked by a GUI to choose the corresponding CSV from which the graph will be build. 
    
    """
    correlation.corel()

@app.command()
def grid_search():
    """ This command will print the best parameters for the Deciscion Tree and Random forest models. 
    
    Args:
        You'll be asked by a GUI to choose the corresponding CSV from which the models will get fitted.
    
    """
    gridsearch.grid()

@app.command()
def pre_processing():
    """This command will start the preprocessing training data. 
    The files must be named traindata_(number).tar.gz . F.e. traindata_1.tar.gz
    It looks for 5 files, doesn't take less or more than that.
    It will save the converted files as features.csv in output/

    Args:
        You'll be asked by a GUI to choose the folder from where the zipped files can be accessed.
    
    """
    preprocessing.pre_pro()

@app.command()
def pre_processing_eval():
    """This command will start the preprocessing for the test data.
    The files must be named testdata_(number).tar.gz . F.e. testdata_1.tar.gz
    It looks for 5 files, doesn't take less or more than that.
    It will save the converted files as eval.csv in output/

    Args:
        You'll be asked by a GUI to choose the folder from where the zipped files can be accessed.
    
    """
    preprocessing_eval.pre_eval()

@app.command()
def votingclassifier():
    """This command will start the training of the voting classifier.
    The classifiers parameters are HARDCODED and can only be changed via editing the voting_classifier.py file!
    Once the model is build it will print a F1-score.
    The command will save the trained model to output/voting_clf.sav .
    With the saved model you'll be able to appply the model to a test dataset.

    The Voting classifier model consists of:
    Decision Tree
    Random Forest
    Gaussian Process Classifier

    All of the above classifiers are weighted by 1.

    Training the model will take some time!



    Args:
        You'll be asked by a GUI to choose the features.csv file for training the model.
    
    """
    voting_classifier.classy()

if __name__ == '__main__':
    app()