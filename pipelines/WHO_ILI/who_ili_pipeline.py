import dlt
import pandas as pd
from typing import Literal
FLUID_URL = "https://xmart-api-public.who.int/FLUMART/VIW_FID_EPI?$format=csv"
FLUNET_URL = "https://xmart-api-public.who.int/FLUMART/VIW_FNT?$format=csv"

def load_who_flu_data(url: str):
    df = pd.read_csv(url)
    df['ISO_WEEKSTARTDATE'] = pd.to_datetime(df.ISO_WEEKSTARTDATE).dt.date
    return df

def run(dataset_id: Literal['fluid', 'flunet'], verbose = True ):
    
    url = {
        'fluid':FLUID_URL,
        'flunet':FLUNET_URL
    }[dataset_id]
    
    # Initialize dlt pipeline
    pipeline = dlt.pipeline(
        pipeline_name="who_ili",
        destination="bigquery",
        dataset_name="case_data"
    )
    
    df = load_who_flu_data(url)
    
    if dataset_id == "flunet":
        df = df.drop(['BVIC_DELUNK', 'BYAM'], axis = 1)
    
    load_info = pipeline.run(
        df,
        table_name="who_" + dataset_id,
        write_disposition="merge",
        primary_key=("COUNTRY_AREA_TERRITORY", "ISO_WEEKSTARTDATE")
    )

    if verbose:
        print(load_info)

if __name__ == "__main__":
    import fire
    fire.Fire(run)