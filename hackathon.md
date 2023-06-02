# PyHealth Sunstella Hackathon 2023 (Optional)
We will spend one week in **June - July** for the pyhealth hackathon. Doing well in this hackathon will:
- strengthen your experience on open-source software development
- get to know pyhealth and learn how to build a typical healthcare AI pipeline
- enrich your engagement in the summer camp (more to mention in your recommendation letter)


| Timeline | Instruction |
| ------ | ------ |
| Before (Now - June 25) | Students will first familiarize themselves with the pyhealth codebase. |
| During (June 25 - July 1) | They select from the existing [spreadsheet](https://docs.google.com/spreadsheets/d/1rSXBT5IJO3Cy0ZacWCg4qC3vvMebida32SuZm2A4qNg/edit#gid=0) to work on one or more contribution items (each item is paired with a mentor). We also highly welcome novel contributions out of the list. The estimated time of each item varies from 1-2 hours to 3-4 days or more. |
| After (July 1 - July 3) | We evaluate the contributions based on the quality of the PRs (one contribution item, one PR). Your efforts here can lead to a strong recommendation letter.|


## Mentors (Core development team)
- `Chaoqi Yang` is a Ph.D. student in CS @ UIUC. His research is build ML models and systems for time-dependent data in health, e.g., electronic health records (EHR), EEG.
- `Zhenbang Wu` is a Ph.D. student in CS @ UIUC. His research interest is in developing generalizable and adaptable deep learning algorithms to solve important healthcare problems.
- `Patrick Jiang` is an M.S. student in CS @ UIUC. His research interest is healthcare natural language processing.
- `Zhen Lin` is a Ph.D. student in CS @ UIUC. His research interests include uncertainty quantification in healthcare and biosignal modeling.
- `Benjamin Danek` is an MCS student in CS @ UIUC. His interests are in federated learning and fairness, and synthetic data generation.
- `Junyi Gao` is a Ph.D. student at the University of Edinburgh funded by the HDR UK-Turing Welcome Ph.D. Program. His research interests include spatio-temporal epidemiology prediction and individual-level clinical predictive modeling.

## Projects 

#### Type 1
We have a list of contribution items in this [spreadsheet](https://docs.google.com/spreadsheets/d/1rSXBT5IJO3Cy0ZacWCg4qC3vvMebida32SuZm2A4qNg/edit#gid=0). Student can put their name in the "Camp Student" column to claim the item. 
> Most of the items are isolated, which asks to add new medical datasets, new AI models, new evaluation metrics, or new healthcare tasks. 

#### Type 2
Students can append new contribution items at the end of the list. Create a new item and leave the "Mentor" part blank (we will assign a mentor for you, or you can pair up with your camp research mentor). For example, use
- your previous related projects
- your summer research projects

## How to frame the PR
Adding new datasets and tasks can refer to [this example](https://github.com/sunlabuiuc/PyHealth/pull/162). Adding new models can refer to [this example](https://github.com/sunlabuiuc/PyHealth/pull/61). The PR is pushed to the `master` branch.

1. Python class follows the structure of `SHHSDataset` in `pyhealth/datasets/shhs.py`
    - with detailed docstring
        ```
        Description: xxx

        Args:
            dataset_name: name of the dataset.
            root: root directory of the raw data (should contain many csv files).
            ...

        Attributes:
            task: Optional[str], name of the task (e.g., "sleep staging").
                Default is None.
            ...

        Examples:
            >>> from pyhealth.datasets import SHHSDataset
            >>> dataset = SHHSDataset(
            ...         root="/srv/local/data/SHHS/",
            ...     )
            >>> dataset.stat()
            >>> dataset.info()
        ```
    - with a runnable example at the end of the file, so that we can use `python pyhealth/datasets/shhs.py` for small data test
        ```
        if __name__ == "__main__":
        dataset = SHHSDataset(
            root="/srv/local/data/SHHS/polysomnography",
            dev=True,
            refresh_cache=True,
        )
        ```

2. Python function follows the structure of `sleep_staging_shhs_fn` in `pyhealth/tasks/sleep_staging.py`
    - with detailed docstring
        ```
        Descriptions: xxx

        Args:
            ...

        Returns:
            ...

        Examples:
            ...
        ```
    - with a runnable example at the end of the file, so that we can use `python pyhealth/datasets/shhs.py` for small data test
        ```
        if __name__ == "__main__":
            from pyhealth.datasets import SHHSDataset

            dataset = SHHSDataset(
                root="/srv/local/data/SHHS/polysomnography",
                dev=True,
                refresh_cache=True,
            )
            sleep_staging_ds = dataset.set_task(sleep_staging_shhs_fn)
            print(sleep_staging_ds.samples[0])
            print(sleep_staging_ds.input_info)
        ```

3. Every PR must has an `example/xxx.py` for large scale data test. For example, `examples/sleep_staging_shhs_contrawr.py` for this case.

