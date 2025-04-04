

# TODOs

## Official data

- [ ] US data parsing

## Data engineering

- kestra docker

``` mermaid
graph LR
    subgraph kestra 
        dlt(dlt) --- posts
        llm --- bqstaging
        llm -- annotation --> bqstaging
        posts --> bqstaging[<b>GBQ</b> \n stage area \n 1 table per kw]
        dlt -- housekeeping --> count
        dlt -- case data --> who_tables
        dlt -- case data --> cdc_tables
        subgraph BigQuery data lake
          bqstaging
          who_tables
          cdc_tables
          count[post counts table]
        end
        bqstaging --- dbt
        who_tables --- dbt
        cdc_tables --- dbt
        count --- dbt
        dbt --> bq[Google \n BigQuery]
        subgraph BigQuery data warehoue
          bq
        end
    end

    bsky[bsky API] --> dlt
    WHO --> dlt
    CDC --> dlt
    bq --> looker[Looker studio \n dashboard]
    bq -- python --> stat1[Statistical analysis]
    bq -- python --> stat2[Machine learning, modeling]
```

- modeling as batch processing pipeline
- run once per day at midnight for the preceding day
- for each data ingestion step a `dlt` pipeline is created which has an
  equivalent `kestra` flow

### `dlt` pipelines

- one pipeline run should extract posts/counts for a single day
- kestra backfills will be used to retrieve historical data
- thus, dlt incremental load can be used

### `kestra` flows

#### Post counts

- [ ] “backfill” flow that can retrieve post counts for a given query
  keywords over a date range
- [ ] “triggered” flow that extracts the post counts for single day and
  will be triggered daily for continous data collection

#### Pull messages

- [ ] “backfill” flow that can retrieve posts for a given query keyword
  over a date range
- [ ] “triggered” flow that extracts the post counts for single day and
  will be triggered daily for continous data collection

#### LLM symptom extraction

- [x] this pipeline/flow should exclude posts that alread have been
  treated

- [ ] handle edge case of no message to label

- case data:

  - WHO
    - [ ] flunet
    - [ ] fluid
  - [ ] CDC
    - gives only weeks –\> date parsing
  - sentinelles ???

### `dbt` models

- per language, per day housekeeping activity

- per day ILI keyword msg counts:

  - long format:A
    - day (index)
    - language
    - keyword
    - count

- per language, per day LLM identified

  - long format:
    - day (index)
    - count ili_related
