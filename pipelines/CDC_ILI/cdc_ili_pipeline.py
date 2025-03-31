import dlt
import pandas as pd
from typing import Literal
from datetime import datetime
from epidatpy import EpiDataContext as Epidata
from epiweeks import Week
import pandas as pd
import fire

def get_epiweek_from_date(input_date: str) -> int:
    """Convert a date to CDC epidemiological week."""
    
    input_date = datetime.fromisoformat(input_date)
    
    return Week.fromdate(input_date)#.cdcweek

def fetch_flu_data(data_type: str) -> pd.DataFrame:
    """
    Fetch flu data from Delphi Epidata API based on the specified type.
    
    Parameters:
        data_type (str): Either "fluview" or "fluview clinical".
        date (datetime): Date for which to fetch data.
        
    Returns:
        pd.DataFrame: DataFrame containing the fetched data.
    """
    # Validate data_type input
    if data_type not in ["fluview", "fluview_clinical"]:
        raise ValueError("Invalid data type. Choose either 'fluview' or 'fluview_clinical'.")
    
    # Fetch data based on the specified type
    if data_type == "fluview":
        response = Epidata().pub_fluview(['nat']).df()
    elif data_type == "fluview_clinical":
        response = Epidata().pub_fluview_clinical(['nat']).df()

    # returnDataFrame
    return response

def run(
    dataset_id: Literal['fluview', 'fluview_clinical'],
    verbose: bool = True
):
    
    # Initialize dlt pipeline
    pipeline = dlt.pipeline(
        pipeline_name="cdc_ili",
        destination="bigquery",
        dataset_name="case_data"
    )
    
    load_info = pipeline.run(
        fetch_flu_data(dataset_id),
        table_name="cdc_" + dataset_id,
        write_disposition="replace"
    )

    if verbose:
        print(load_info)
        
        
if __name__ == "__main__":
    fire.Fire(run)