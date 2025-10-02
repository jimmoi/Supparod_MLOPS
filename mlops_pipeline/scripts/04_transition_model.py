import sys
from mlflow.tracking import MlflowClient


def transition_model_alias(model_name, alias):
    """
    Sets an alias for the latest version of a registered model.
    This replaces the deprecated "stage" functionality.
    """
    client = MlflowClient()
    try:
        # Find the latest version of the model by searching and sorting
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            print(f"Error: No versions found for model '{model_name}'.")
            sys.exit(1)
            
        latest_version = max(versions, key=lambda mv: int(mv.version))
        version_number = latest_version.version
        print(f"Found latest model version: {version_number}")


        # Set the alias for the latest version
        # This is the modern replacement for transitioning stages
        print(f"Setting alias '{alias}' for model version {version_number}...")
        client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=version_number
        )
        print(f"Successfully set alias '{alias}' for model '{model_name}' version {version_number}.")


    except Exception as e:
        print(f"An error occurred while setting the model alias: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/04_transition_model.py <model_name> <alias>")
        sys.exit(1)
    
    model_name_arg = sys.argv[1]
    target_alias_arg = sys.argv[2]
    transition_model_alias(model_name_arg, target_alias_arg)


