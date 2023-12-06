from mlflow import MlflowClient # enables command line mgmt of experiments
from pprint import pprint # makes things look nice
from sklearn.ensemble import RandomForestRegressor # an ML model
from apple_generator import generate_apple_sales_data_with_promo_adjustment
import numpy as np


# Connect to an existing MLFlow database, same URI as other scripts
# The activation of the server is done elsewhere / earlier
client = MlflowClient(tracking_uri="http://127.0.0.1:8080")


# The MLFlow Client import enables searching of existing experiments
# Search experiments without providing query terms behaves effectively as a 'list' action

all_experiments = client.search_experiments()

pprint(all_experiments)

# Extract the experiment name and lifecycle_stage

default_experiment = [
    {"name": experiment.name, "lifecycle_stage": experiment.lifecycle_stage}
    for experiment in all_experiments
    if experiment.name == "Default"
][0]

pprint(default_experiment)

# THIS IS WHERE THE EXPERIMENT IS CREATED

# This is shown in the UI
experiment_description = (
    "This is the grocery forecasting project. "
    "This experiment contains the produce models for apples."
)

# Not currently visible in UI?
# Seems domain-specific, who is the audience?
experiment_tags = {
    "project_name": "grocery-forecasting",
    "store_dept": "produce",
    "team": "stores-ml",
    "project_quarter": "Q3-2023",
    "mlflow.note.content": experiment_description,
}


# So this is the experiment itself, created via clien.create_experiment(<details>)
# remember that 'client' is MlflowClient(<URI>)
# also, this is where it stops due to duplicate experiment names
# how are experiment names updated?
# alternatively, how are 'runs' within 'experiments' initiated?

# COMMENTED THIS OUT AS THE EXPERIMENT IS DEFINED THE FIRST TIME THIS IS RUN

# produce_apples_experiment = client.create_experiment(name="Apple_Models_3", tags=experiment_tags)

# Use search_experiments() to search on the project_name tag key
# what is going on with the backticks here?

apples_experiment = client.search_experiments(
    filter_string="tags.`project_name` = 'grocery-forecasting'"
)

pprint(apples_experiment[0])

# Access individual tag data

print(apples_experiment[0].tags["team"])

# Calls the imported function
data = generate_apple_sales_data_with_promo_adjustment(base_demand=1_000, n_rows=1_000)

# THIS IS WHERE THE MODEL IS IMPORTED AND TRAINED

import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Use the fluent API to set the tracking uri and the active experiment
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# Sets the current active experiment to the "Apple_Models" experiment and returns the Experiment metadata

# So this is how more info (ie, runs) can be added to an existing experiment: set_experiment
# So I could create an experiment, empty, in the UI, then set it as the experiment here
apple_experiment = mlflow.set_experiment("Apple_Models")

# Define a run name for this iteration of training.
# If this is not set, a unique name will be auto-generated for your run.
run_name = "apples_rf_test_2"

# Define an artifact path that the model will be saved to.
artifact_path = "rf_apples"


# Split the data into features and target and drop irrelevant date field and target field
X = data.drop(columns=["date", "demand"])
y = data["demand"]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

params = {
    "n_estimators": 100,
    "max_depth": 6,
    "min_samples_split": 10,
    "min_samples_leaf": 4,
    "bootstrap": True,
    "oob_score": False,
    "random_state": 888,
}

# Train the RandomForestRegressor
rf = RandomForestRegressor(**params)

# Fit the model on the training data
rf.fit(X_train, y_train)

# Predict on the validation set
y_pred = rf.predict(X_val)

# Calculate error metrics
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred)

# Assemble the metrics we're going to write into a collection
metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

# Initiate the MLflow run context
with mlflow.start_run(run_name=run_name) as run:
    # Log the parameters used for the model fit
    mlflow.log_params(params)

    # Log the error metrics that were calculated during validation
    mlflow.log_metrics(metrics)

    # Log an instance of the trained model for later use
    mlflow.sklearn.log_model(sk_model=rf, input_example=X_val, artifact_path=artifact_path)