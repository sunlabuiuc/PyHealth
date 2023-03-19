---
layout: default
title: "PyHealth: A Deep Learning Toolkit For Healthcare Applications"
permalink: /



Authors:

  - name: Chaoqi Yang
    url: https://ycq091044.github.io/
    aff: PhD@UIUC
    image: assets/images/chaoqi.jpg
    email: chaoqiy2@illinois.edu

  - name: Zhenbang Wu
    url: https://zzachw.github.io/
    aff: PhD@UIUC
    image: assets/images/zhenbang.jpg
    email: zw12@illinois.edu

  - name: Patrick Jiang
    url: https://www.linkedin.com/in/patrick-j-3492b4235/
    aff: MS@UIUC
    image: assets/images/patrick.jpg
    email: pj20@illinois.edu

  - name: Zhen Lin
    url: https://zlin7.github.io/
    aff: PhD@UIUC
    image: assets/images/zhen.png 
    email: zhenlin4@illinois.edu

  - name: Junyi Gao
    url: http://aboutme.vixerunt.org/
    aff: PhD@University of Edinburgh
    image: assets/images/junyi.jpg 
    email: junyi.gao@ed.ac.uk

  - name: Bejamin Danek
    url: https://bpdanek.github.io/
    aff: SDE@New Relic, MS@UIUC
    image: assets/images/bejamin.jpg 
    email: danekbenjamin@gmail.com

  - name: Jimeng Sun
    url: https://sunlab.org/
    aff: CS Professor@UIUC
    image: assets/images/jimeng.png 
    email: jimeng.sun@gmail.com


---


<img src="../figure/poster.png" width="620">

## 1. Installation :rocket:

- You could install from PyPi:
```
pip install pyhealth
```
- or from github source:
```
pip install .
```

## 2. Introduction :book:
``pyhealth`` provides these functionalities (we are still enriching some modules):

<img src="../figure/overview.png" width="770">

You can use the following functions independently:

- **Dataset**: ``MIMIC-III``, ``MIMIC-IV``, ``eICU``, ``OMOP-CDM``, ``customized EHR datasets``, etc.
- **Tasks**: ``diagnosis-based drug recommendation``, ``patient hospitalization and mortality prediction``, ``length stay forecasting``, etc. 
- **ML models**: ``CNN``, ``LSTM``, ``GRU``, ``LSTM``, ``RETAIN``, ``SafeDrug``, ``Deepr``, etc.

*Build a healthcare AI pipeline can be as short as 10 lines of code in PyHealth*.


## 3. Build ML Pipelines :trophy:

All healthcare tasks in our package follow a **five-stage pipeline**: 
<img src="../figure/five-stage-pipeline.png" width="640">

 We try hard to make sure each stage is as separate as possibe, so that people can customize their own pipeline by only using our data processing steps or the ML models.



## 4. Medical Code Map :hospital: 

``pyhealth.codemap`` provides two core functionalities. **This module can be used independently.**

* For code ontology lookup within one medical coding system (e.g., name, category, sub-concept); 

```python
from pyhealth.medcode import InnerMap

icd9cm = InnerMap.load("ICD9CM")
icd9cm.lookup("428.0")
# `Congestive heart failure, unspecified`

atc = InnerMap.load("ATC")
atc.lookup("M01AE51")
# `ibuprofen, combinations`
atc.lookup("M01AE51", "description")
# Ibuprofen is a non-steroidal anti-inflammatory drug (NSAID) derived ...
```

* For code mapping between two coding systems (e.g., ICD9CM to CCSCM). 
```python
from pyhealth.medcode import CrossMap

codemap = CrossMap.load("ICD9CM", "CCSCM")
codemap.map("428.0")
# ['108']
```

## 5. Medical Code Tokenizer :speech_balloon:

``pyhealth.tokenizer`` is used for transformations between string-based tokens and integer-based indices, based on the overall token space. We provide flexible functions to tokenize 1D, 2D and 3D lists. **This module can be used independently.**
```python
from pyhealth.tokenizer import Tokenizer

# Example: we use a list of ATC3 code as the token
token_space = ['A01A', 'A02A', 'A02B', 'A02X', 'A03A', 'A03B', 'A03C', 'A03D', \
        'A03F', 'A04A', 'A05A', 'A05B', 'A05C', 'A06A', 'A07A', 'A07B', 'A07C', \
        'A12B', 'A12C', 'A13A', 'A14A', 'A14B', 'A16A']
tokenizer = Tokenizer(tokens=token_space, special_tokens=["<pad>", "<unk>"])

# 2d encode & decode
...

# 3d encode & decode
...
```

## 6. Tutorials :teacher:

We provide the following tutorials to help users get started with our pyhealth. 

- [Tutorial 0: Introduction to pyhealth.data](https://colab.research.google.com/drive/1y9PawgSbyMbSSMw1dpfwtooH7qzOEYdN?usp=sharing) [[Video]](https://www.youtube.com/watch?v=Nk1itBoLOX8&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=2)  

- [Tutorial 1: Introduction to pyhealth.datasets](https://colab.research.google.com/drive/18kbzEQAj1FMs_J9rTGX8eCoxnWdx4Ltn?usp=sharing) [[Video]](https://www.youtube.com/watch?v=c1InKqFJbsI&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=3)  

- [Tutorial 2: Introduction to pyhealth.tasks](https://colab.research.google.com/drive/1r7MYQR_5yCJGpK_9I9-A10HmpupZuIN-?usp=sharing) [[Video]](https://www.youtube.com/watch?v=CxESe1gYWU4&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=4) 

- [Tutorial 3: Introduction to pyhealth.models](https://colab.research.google.com/drive/1LcXZlu7ZUuqepf269X3FhXuhHeRvaJX5?usp=sharing) [[Video]](https://www.youtube.com/watch?v=fRc0ncbTgZA&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=6)  

- [Tutorial 4: Introduction to pyhealth.trainer](https://colab.research.google.com/drive/1L1Nz76cRNB7wTp5Pz_4Vp4N2eRZ9R6xl?usp=sharing) [[Video]](https://www.youtube.com/watch?v=5Hyw3of5pO4&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=7)  

- [Tutorial 5: Introduction to pyhealth.metrics](https://colab.research.google.com/drive/1Mrs77EJ92HwMgDaElJ_CBXbi4iABZBeo?usp=sharing) [[Video]](https://www.youtube.com/watch?v=d-Kx_xCwre4&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=8) 


- [Tutorial 6: Introduction to pyhealth.tokenizer](https://colab.research.google.com/drive/1bDOb0A5g0umBjtz8NIp4wqye7taJ03D0?usp=sharing) [[Video]](https://www.youtube.com/watch?v=CeXJtf0lfs0&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=10) 


- [Tutorial 7: Introduction to pyhealth.medcode](https://colab.research.google.com/drive/1xrp_ACM2_Hg5Wxzj0SKKKgZfMY0WwEj3?usp=sharing) [[Video]](https://www.youtube.com/watch?v=MmmfU6_xkYg&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=9)


The following tutorials will help users build their own task pipelines.

- [Pipeline 1: Drug Recommendation](https://colab.research.google.com/drive/10CSb4F4llYJvv42yTUiRmvSZdoEsbmFF?usp=sharing)

- [Pipeline 2: Length of Stay Prediction](https://colab.research.google.com/drive/1JoPpXqqB1_lGF1XscBOsDHMLtgvlOYI1?usp=sharing)

- [Pipeline 3: Readmission Prediction](https://colab.research.google.com/drive/1bhCwbXce1YFtVaQLsOt4FcyZJ1_my7Cs?usp=sharing)

- [Pipeline 4: Mortality Prediction](https://colab.research.google.com/drive/1Qblpcv4NWjrnADT66TjBcNwOe8x6wU4c?usp=sharing)

- [Pipeline 5: Sleep Staging](https://colab.research.google.com/drive/1mpSeNCAthXG3cqROkdUcUdozIPIMTCuo?usp=sharing)


 We provided the advanced tutorials for supporting various needs. 

- [Advanced Tutorial 1: Fit your dataset into our pipeline](https://colab.research.google.com/drive/1UurxwAAov1bL_5OO3gQJ4gAa_paeJwJp?usp=sharing) [[Video]](https://www.youtube.com/watch?v=xw2hGLEQ4Y0&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=13)

- [Advanced Tutorial 2: Define your own healthcare task](https://colab.research.google.com/drive/1gK6zPXvfFGBM1uNaLP32BOKrnnJdqRq2?usp=sharing) 

- [Advanced Tutorial 3: Adopt customized model into pyhealth](https://colab.research.google.com/drive/1F_NJ90GC8_Eq-vKTf7Tyziew4gWjjKoH?usp=sharing) [[Video]](https://www.youtube.com/watch?v=lADFlcmLtdE&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=14)

- [Advanced Tutorial 4: Load your own processed data into pyhealth and try out our ML models](https://colab.research.google.com/drive/1ZRnKch2EyJLrI3G5AvDXVpeE2wwgBWfw?usp=sharing) [[Video]](https://www.youtube.com/watch?v=xw2hGLEQ4Y0&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=13)


## 7. Datasets :mountain_snow:
We provide the processing files for the following open EHR datasets:

| Dataset   | Module                                    | Year | Information                                                                                                     |
| --------- | ----------------------------------------- | ---- | -------------------------------------------------------------------------------------------------------------- |
| MIMIC-III | `pyhealth.datasets.MIMIC3Dataset`         | 2016 | [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4//)                                  |
| MIMIC-IV  | `pyhealth.datasets.MIMIC4Dataset`         | 2020 | [MIMIC-IV Clinical Database](https://physionet.org/content/mimiciv/0.4/)                                       |
| eICU      | `pyhealth.datasets.eICUDataset`           | 2018 | [eICU Collaborative Research Database](https://eicu-crd.mit.edu//)                                          |
| OMOP      | `pyhealth.datasets.OMOPDataset`           |      | [OMOP-CDM schema based dataset](https://www.ohdsi.org/data-standardization/the-common-data-model)  |
| SleepEDF  | `pyhealth.datasets.SleepEDFDataset`       | 2018 | [Sleep-EDF dataset](https://physionet.org/content/sleep-edfx/1.0.0/)                                            |
| SHHS      | `pyhealth.datasets.SHHSDataset`           | 2016 | [Sleep Heart Health Study dataset](https://sleepdata.org/datasets/shhs)                                         |
| ISRUC     | `pyhealth.datasets.ISRUCDataset`          | 2016 | [ISRUC-SLEEP dataset](https://sleeptight.isr.uc.pt/?page_id=48)                                                 |

## 8. Tutorial Schedule :mortar_board:
TBA

## 9. Authors :information_desk_person:

{% include team.html id="Authors" %}


















