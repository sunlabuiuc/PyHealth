# PyHealth OMOP Development PLAN

### Step 1: Build OMOP data base
- find existing OMOP mapping and integrate into our pipeline
    - for example https://github.com/MIT-LCP/mimic-omop

### Step 2: Customize data API
- based on the OMOP data, we provide APIs for querying the following features
    - sequence features
    - demographics features
    - indicators features (mortality, readmission)
    - others

### Step 3: Model Template
- provide a running deep learning model on using the OMOP data with API, so that contributors can refer to the template and pr their own models.
