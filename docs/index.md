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


<img src="assets/images/poster.png" style="border-radius: 0%;" width="720">

PyHealth is designed for both **ML researchers and medical practitioners**. We can make your **healthcare AI applications** easier to deploy, test and validate. Your development process becomes more flexible and more customizable.

```
pip install pyhealth
```

## 1. Introduction
``pyhealth`` provides these functionalities (we are still enriching some modules):

<img src="assets/images/overview.png" style="border-radius: 0%;"  width="770">

You can use the following functions independently:

- **Dataset**: ``MIMIC-III``, ``MIMIC-IV``, ``eICU``, ``OMOP-CDM``, ``Sleep-EDF``, ``Chest X-ray``, ``ISURC``, etc.
- **Tasks**: ``diagnosis-based drug recommendation``, ``patient hospitalization and mortality prediction``, ``sleep staging``, ``chest disease classificatio``, etc. 
- **ML models**: ``CNN``, ``LSTM``, ``GRU``, ``LSTM``, ``RETAIN``, ``SafeDrug``, ``Deepr``, ``SparcNet``, ``ContraWR``, ``GAMENet``, etc.

*Build a healthcare AI pipeline can be as short as 10 lines of code in PyHealth*.


## 2. Build ML Pipelines

All healthcare tasks in our package follow a **five-stage pipeline**: 
<img src="assets/images/five-stage-pipeline.png" style="border-radius: 0%;"  width="640">

We try hard to ensure the modules are independent, so that people can customize their own pipeline by only using parts of our pipelines, such as using data processing steps or borrowing the ML models.



## 3. Medical Code Map

``pyhealth.codemap`` provides two core functionalities. **This module can be used independently.**

* For code ontology lookup within one medical coding system (e.g., name, category, sub-concept); 

```python
from pyhealth.medcode import InnerMap

icd9cm = InnerMap.load("ICD9CM")
icd9cm.lookup("428.0")
# `Congestive heart failure, unspecified`

atc = InnerMap.load("ATC")
atc.lookup("M01AE51") # `ibuprofen, combinations`
atc.lookup("M01AE51", "description")
# Ibuprofen is a non-steroidal anti-inflammatory drug (NSAID) derived ...
```

* For code mapping between two coding systems (e.g., ICD9CM to CCSCM). 

```python
from pyhealth.medcode import CrossMap

codemap = CrossMap.load("ICD9CM", "CCSCM")
codemap.map("428.0") # ['108']
```

## 4. Medical Code Tokenizer

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

## 5. Colab Tutorials

We provide the following colab tutorials to help users get started with our pyhealth. 

- [Tutorial 0: Introduction to pyhealth.data](https://colab.research.google.com/drive/1y9PawgSbyMbSSMw1dpfwtooH7qzOEYdN?usp=sharing) [[Video]](https://www.youtube.com/watch?v=Nk1itBoLOX8&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=2)  

- [Tutorial 1: Introduction to pyhealth.datasets](https://colab.research.google.com/drive/18kbzEQAj1FMs_J9rTGX8eCoxnWdx4Ltn?usp=sharing) [[Video]](https://www.youtube.com/watch?v=c1InKqFJbsI&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=3)  

- [Tutorial 2: Introduction to pyhealth.tasks](https://colab.research.google.com/drive/1r7MYQR_5yCJGpK_9I9-A10HmpupZuIN-?usp=sharing) [[Video]](https://www.youtube.com/watch?v=CxESe1gYWU4&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=4) 

- [Tutorial 3: Introduction to pyhealth.models](https://colab.research.google.com/drive/1LcXZlu7ZUuqepf269X3FhXuhHeRvaJX5?usp=sharing) [[Video]](https://www.youtube.com/watch?v=fRc0ncbTgZA&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=6)  

- [Tutorial 4: Introduction to pyhealth.trainer](https://colab.research.google.com/drive/1L1Nz76cRNB7wTp5Pz_4Vp4N2eRZ9R6xl?usp=sharing) [[Video]](https://www.youtube.com/watch?v=5Hyw3of5pO4&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=7)  

- [Tutorial 5: Introduction to pyhealth.metrics](https://colab.research.google.com/drive/1Mrs77EJ92HwMgDaElJ_CBXbi4iABZBeo?usp=sharing) [[Video]](https://www.youtube.com/watch?v=d-Kx_xCwre4&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=8) 


- [Tutorial 6: Introduction to pyhealth.tokenizer](https://colab.research.google.com/drive/1bDOb0A5g0umBjtz8NIp4wqye7taJ03D0?usp=sharing) [[Video]](https://www.youtube.com/watch?v=CeXJtf0lfs0&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=10) 


- [Tutorial 7: Introduction to pyhealth.medcode](https://colab.research.google.com/drive/1xrp_ACM2_Hg5Wxzj0SKKKgZfMY0WwEj3?usp=sharing) [[Video]](https://www.youtube.com/watch?v=MmmfU6_xkYg&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=9)


The following colab tutorials will help users build their own task pipelines.

- [Pipeline 1: Drug Recommendation](https://colab.research.google.com/drive/10CSb4F4llYJvv42yTUiRmvSZdoEsbmFF?usp=sharing) [[Video]](https://www.youtube.com/watch?v=GGP3Dhfyisc&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=12)

- [Pipeline 2: Length of Stay Prediction](https://colab.research.google.com/drive/1JoPpXqqB1_lGF1XscBOsDHMLtgvlOYI1?usp=sharing) [[Video]](https://www.youtube.com/watch?v=GGP3Dhfyisc&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=12)

- [Pipeline 3: Readmission Prediction](https://colab.research.google.com/drive/1bhCwbXce1YFtVaQLsOt4FcyZJ1_my7Cs?usp=sharing) [[Video]](https://www.youtube.com/watch?v=GGP3Dhfyisc&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=12)

- [Pipeline 4: Mortality Prediction](https://colab.research.google.com/drive/1Qblpcv4NWjrnADT66TjBcNwOe8x6wU4c?usp=sharing) [[Video]](https://www.youtube.com/watch?v=GGP3Dhfyisc&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=12)

- [Pipeline 5: Sleep Staging](https://colab.research.google.com/drive/1mpSeNCAthXG3cqROkdUcUdozIPIMTCuo?usp=sharing) [[Video]](https://www.youtube.com/watch?v=ySAIU-rO6so&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=16)


 We provided the advanced colab tutorials for supporting various needs. 

- [Advanced Tutorial 1: Fit your dataset into our pipeline](https://colab.research.google.com/drive/1UurxwAAov1bL_5OO3gQJ4gAa_paeJwJp?usp=sharing) [[Video]](https://www.youtube.com/watch?v=xw2hGLEQ4Y0&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=13)

- [Advanced Tutorial 2: Define your own healthcare task](https://colab.research.google.com/drive/1gK6zPXvfFGBM1uNaLP32BOKrnnJdqRq2?usp=sharing) 

- [Advanced Tutorial 3: Adopt customized model into pyhealth](https://colab.research.google.com/drive/1F_NJ90GC8_Eq-vKTf7Tyziew4gWjjKoH?usp=sharing) [[Video]](https://www.youtube.com/watch?v=lADFlcmLtdE&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=14)

- [Advanced Tutorial 4: Load your own processed data into pyhealth and try out our ML models](https://colab.research.google.com/drive/1ZRnKch2EyJLrI3G5AvDXVpeE2wwgBWfw?usp=sharing) [[Video]](https://www.youtube.com/watch?v=xw2hGLEQ4Y0&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=13)

## 6. KDD Tutorial Schedule
We will release the slides and colab notebooks on [Google Drive](https://drive.google.com/drive/folders/10SRErhMgmwIvBwafp_YmaZEziOhYTaYk?usp=sharing) before the tutorial.
### 6.1 Introduction to PyHealth (20 min)
We will use slides to present this part.
- Background and motivations.
- Key features of PyHealth.
- Quickstart examples.

### 6.2 PyHealth for EHR (40 min)
We will use slides as well as colab notebooks to present.
- Introduce the five-stage pipeline
- (Stage 1) Load EHR datasets, such as MIMIC-III, MIMIC-IV, eICU, OMOP-CDM
- (Stage 2) Define healthcare tasks, such as drug recommendation, length of stay prediction.
- (Stage 3) Initialize healthcare AI models, such as RETAIN, GAMENet, MICRON, and SafeDrug.
- (Stage 4) Train the model.
- (Stage 5) Evaluate the model.

It is worth noting that the same five-stage pipeline applies to other data modalities as well. Therefore, we will focus on introducing the datasets, tasks, and models in the subsequent sections.

### 6.3 PyHealth for physiological signals (20 min)
We will use slides as well as colab notebooks to present.
- Overview of biosignal datasets supported by PyHealth, such as ISRUC, Sleep-EDF, and SHHS.
- Introduce existing biosignal models in PyHealth.
- Demonstrate the sleep staging task on the Sleep-EDF dataset using SparcNet.

### 6.4 PyHealth for medical imaging (15 min)
We will use slides as well as colab notebooks to present.
- Overview of medical image datasets supported by PyHealth, such as CheXpert, RSNA, COVID, and MIMIC-CXR.
- Overview of tasks supported by PyHealth, such as x-ray representation learning, chest disease classification, medical report generation.
- Introduce existing models in PyHealth.
- Demonstrate the chest disease classification on the COVID dataset using ResNet.

### 6.5 PyHealth for biomedical text mining (15 min)
We will use slides as well as colab notebooks to present.
- Overview of medical text datasets supported by PyHealth, such as MIMIC-III clinical notes, MIMIC-CXR, and IU-XRay.
- Overview of tasks supported by PyHealth, such as clinical notes classification and medical report generation.
- Introduce existing models in PyHealth.
- Demonstrate on radiology reports generation from x-ray images.

### 6.6 PyHealth pre-trained embedding (30 min) 
We will use slides as well as colab notebooks to present.
- Overview of the medical knowledge base, including the supported medical coding systems, code mappings, and 
- Medical concept lookup.
- Medical code mapping.
- Pre-trained medical concepte embeddings.
- Demo: Leveraing the UMLS knowledge graph embedding to improve drug recommendation task on the MIMIC-III dataset.

### 6.7 PyHealth uncertainty quantification and model calibration (30 min)
We will use slides as well as colab notebooks to present.
- Introduction of basic concepts and common post-hoc tasks in uncertainty quantification, such as model calibration,  prediction set construction and prediction interval construction. 
- How to perform such tasks on arbitrary PyHealth models, after the training is done (which is why this is post-hoc).
- (Demo 1) We will use Temperature Scaling, Historgram Binning, and Kernel-based Calibration to calibrate a trained SPaRCNet, used for sleep staging task on the ISRUC dataset. (show some plots to let user understand the concepts)
- (Demo 2) We will use conformal prediction (LABEL) to construct prediction set with guarantee on the mis-coverage risk, again on a trained SPaRCNet on sleep staing task with the ISRUC dataset. 

### 6.8 Conclusion (10 min)
- A brief summary of the benefits of using PyHealth.
- Future development plans for PyHealth.
- A call for contributions from the community.
- Useful resources for PyHealth users.
## 7. PyHealth Tutors

{% include team.html id="Authors" %}


















