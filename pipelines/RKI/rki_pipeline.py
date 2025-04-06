import dlt
import pandas as pd
from typing import Literal
from datetime import datetime, timedelta

GRIPPENET_URL = 'https://raw.githubusercontent.com/robert-koch-institut/GrippeWeb_Daten_des_Wochenberichts/refs/heads/main/GrippeWeb_Daten_des_Wochenberichts.tsv'
SENTINEL_URL = 'https://raw.githubusercontent.com/robert-koch-institut/ARE-Konsultationsinzidenz/refs/heads/main/ARE-Konsultationsinzidenz.tsv'
HOSPITAL_URL = 'https://raw.githubusercontent.com/robert-koch-institut/SARI-Hospitalisierungsinzidenz/refs/heads/main/SARI-Hospitalisierungsinzidenz.tsv'

def year_week_to_isoweek_start(year_week_str):
  """
  Converts a string of the form YYYY-WNN to the ISO week start date (Monday).

  Args:
    year_week_str: The year and week string, e.g., "2023-W52".

  Returns:
    A datetime.date object representing the ISO week start date (Monday).
    Returns None if the input string is invalid or if the year/week combination
    is invalid.
  """
  try:
    year, week = map(int, year_week_str.split('-W'))
    # Python uses 1-indexed weeks.
    date = datetime.strptime(f"{year}-W{week}-1", "%Y-W%W-%w")
    
    #The strptime function returns the isoweek day, but we want the monday of that week.
    #So get the weekday.  Monday is represented by 0, thus subtract the weekday from the given date
    weekday = date.weekday()
    isoweek_start_date = date - timedelta(days=weekday)

    return isoweek_start_date.date()

  except ValueError as e:
    print(e)
    return None


def load_rki_dataset(url: str):
    df = pd.read_csv(url, sep = '\t')
    df['ISO_WEEKSTARTDATE'] = df.Kalenderwoche.apply(year_week_to_isoweek_start)
    return df

def run(dataset_id: Literal['fluid', 'flunet'], verbose = True ):
    
    url = {
        'grippenet': GRIPPENET_URL,
        'sentinel': SENTINEL_URL,
        'hospital': HOSPITAL_URL
    }[dataset_id]
    
    # Initialize dlt pipeline
    pipeline = dlt.pipeline(
        pipeline_name="rki",
        destination="bigquery",
        dataset_name="case_data"
    )
    
    df = load_rki_dataset(url)
    
    load_info = pipeline.run(
        df,
        table_name="rki_" + dataset_id,
        write_disposition="replace",
    )

    if verbose:
        print(load_info)

if __name__ == "__main__":
    import fire
    fire.Fire(run)