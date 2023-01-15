import typer
import gridsearch
import preprocessing
import preprocessing_eval
import voting_classifier
import correlation

app = typer.Typer(epilog="Team 4 | Max & Max")

@app.command()

def label_test_data():
    
