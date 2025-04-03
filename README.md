

# TODOs

## Official data

- [ ] US data parsing

## Data engineering

- kestra docker

``` mermaid
graph LR
    subgraph kestra 
        dlt(dlt) --- posts
        llm -- annotation --> posts
        posts --> bqstaging[<b>GBQ</b> \n stage area \n 1 table per kw]
        dlt -- housekeeping --> csv
        dlt -- case data --> csv
        subgraph cloud staging
          bqstaging
          csv[<b>csv seeds</b> \n GCS bucket \n or Drive]
        end
        bqstaging --- dbt
        csv --- dbt
        dbt --> bq[Google \n BigQuery]
    end

    bsky[bsky API] --> dlt
    WHO --> dlt
    CDC --> dlt
    bq --> looker[Looker studio \n dashboard]
```

- modeling as batch processing pipeline
- run once per day at midnight for the preceding day

### `dlt` pipelines

- one pipeline run should extract posts/counts for a single day

- kestra backfills will be used to retrieve historical data

- thus, dlt incremental load can be used

- [ ] message extraction and storage -\> write to duckdb

- [ ] message counts based on keywords -\> csv file to `dbt` seed folder

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

  - long format:
    - day (index)
    - language
    - keyword
    - count

- per language, per day LLM identified

  - long format:
    - day (index)
    - count ili_related

#### Seeds

- `.csv` files from housekeeping
- `.csv` files with case data
