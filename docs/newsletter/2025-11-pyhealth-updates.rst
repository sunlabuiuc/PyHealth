November 2025 PyHealth Updates
==============================

*Published: November 17, 2025*

Hey everyone,

The PyHealth development team here with an update on what we've been working on over the past few months. We've made some significant progress on features that many of you have been requesting, and we wanted to share where things stand.

**What's in this update:**

- Memory optimizations that make MIMIC-IV workable on smaller machines
- Interpretability module for clinical predictive models
- Conformal prediction and uncertainty quantification
- Bounty system progress (new datasets and restored models)
- Research initiative updates (published 3 papers at ML4H!)


----


Memory Optimizations: Making MIMIC-IV Accessible
-------------------------------------------------

If you've tried working with large datasets like MIMIC-IV, you've probably run into a major pain point: memory requirements. Loading patient lab and chart events can balloon to over 300GB of RAM, which means you basically need a compute cluster just to get started. This has been a barrier for a lot of researchers who want to work with these datasets but don't have access to that kind of hardware.

We've been tackling this problem with a new streaming mode that leverages Polars' lazy loading capabilities. The key insight is that we don't need to load everything into memory at once—instead, we can read and write patient events from disk on-demand as we process them.

**Here's what we're seeing so far:**

.. list-table::
   :header-rows: 1
   :widths: 30 20 25 25

   * - Method
     - Processing Speed (samples/s)
     - Total Memory (GB)
     - Memory Improvement
   * - Pandas (baseline)
     - 4.92
     - 315.84
     - —
   * - PyHealth 2.0
     - 14.99
     - 315.84
     - —
   * - **PyHealth 2.0 Stream**
     - **11.24**
     - **34.25**
     - **~10x reduction** ⚡

The streaming mode is still a work in progress, but we're getting close. Our goal is to democratize access to large-scale healthcare datasets—if you have a decent workstation, you should be able to train clinical predictive models without needing expensive infrastructure. We think this will open up healthcare AI research to a lot more people.


----


Interpretability Module for Clinical Models
--------------------------------------------

`Documentation → <https://pyhealth.readthedocs.io/en/latest/api/interpret.html>`_

One of the most common requests we get from both researchers and clinicians is: "How do I understand why my model is making these predictions?" Interpretability isn't just nice to have in healthcare—it's often essential for building trust and getting models deployed in clinical settings.

The problem is that most existing interpretability tools weren't built with clinical data in mind. They assume you're working with images or continuous features, not the multimodal, structured clinical data we deal with (diagnosis codes, lab values, procedures, medications, etc.). Plus, they usually only implement one technique and don't give you quantitative ways to evaluate whether your explanations are actually faithful to what the model is doing.

**We've built a new interpretability module that addresses these issues:**

- A flexible API that works with PyHealth models out of the box
- Multiple attribution methods: Integrated Gradients, DeepLift, and Chefer (specifically for transformers)
- Evaluation metrics (comprehensiveness and sufficiency) so you can actually quantify how good your explanations are

**Available Methods:**

.. code-block:: python

   from pyhealth.interpret.methods import (
       IntegratedGradients,
       DeepLift,
       CheferRelevance,  # For transformer models
   )

**Example Usage:**

.. code-block:: python

   from pyhealth.datasets import MIMIC4Dataset
   from pyhealth.models import StageNet
   from pyhealth.interpret.methods import IntegratedGradients
   from pyhealth.metrics.interpretability import evaluate_attribution
   
   # Load dataset and train model
   dataset = MIMIC4Dataset(root="data/", tables=["diagnoses_icd", "labevents"])
   samples = dataset.set_task(task_fn)
   model = StageNet(dataset=samples)
   # ... train model ...
   
   # Compute attributions
   ig = IntegratedGradients(model, use_embeddings=True)
   attributions = ig.attribute(**batch, steps=50)
   
   # Evaluate with metrics
   results = evaluate_attribution(
       model,
       test_loader,
       ig,
       metrics=["comprehensiveness", "sufficiency"],
       percentages=[25, 50, 99]
   )
   print(f"Comprehensiveness: {results['comprehensiveness']:.4f}")
   print(f"Sufficiency: {results['sufficiency']:.4f}")

The module is live now and ready to use. We're continuing to add more interpretability techniques and refine the existing implementations based on feedback. If there's a specific method you'd like to see added, let us know!


----


Uncertainty Quantification with Conformal Prediction
----------------------------------------------------

`Documentation → <https://pyhealth.readthedocs.io/en/latest/api/calib.html>`_

Another major wishlist item we've been working on is uncertainty quantification. In healthcare, it's not enough to just get a prediction—you need to know when your model is uncertain. Traditional calibration helps (it makes your model's confidence scores actually meaningful), but we wanted to go further.

Conformal prediction is a framework that gives you prediction sets with theoretical coverage guarantees. Instead of saying "Disease A with 80% confidence," you can say "The true diagnosis is in {Disease A, Disease B, Disease C}, and we guarantee the right answer is in this set 99% of the time." This gives clinicians much more actionable information, especially when multiple diagnoses are plausible.

**The challenge:** Healthcare data is messy in ways that break standard assumptions. You've got covariate shifts (patient populations change), missing data everywhere, and distributions that violate the I.I.D. assumptions that both ML and conformal prediction typically rely on. Standard uncertainty techniques often fail in these real-world conditions, which is exactly when you need them most.

**We're actively building out tools to handle these challenges:**

- Conformal prediction under covariate shift
- Negative sampling techniques for conformal prediction
- Patient distribution-adaptive conformal prediction

These aren't just theoretical research directions—they're aimed at solving the practical problems you run into when trying to deploy models in actual clinical settings.

**Example: Covariate Shift Adaptive Conformal Prediction**

Here's a preliminary example from our COVID-19 chest X-ray classification work, giving you a sneak peek of what's coming. This shows how to handle distribution shifts between your calibration and test sets:

.. code-block:: python

   from pyhealth.datasets import COVID19CXRDataset, split_by_sample_conformal
   from pyhealth.models import TorchvisionModel
   from pyhealth.calib.predictionset import LABEL
   from pyhealth.calib.predictionset.covariate import CovariateLabel
   from pyhealth.calib.utils import extract_embeddings
   
   # Load dataset and split into train/val/cal/test
   dataset = COVID19CXRDataset(root="data/")
   samples = dataset.set_task()
   train, val, cal, test = split_by_sample_conformal(samples, [0.6, 0.1, 0.15, 0.15])
   
   # Train ResNet-18
   model = TorchvisionModel(dataset=samples, model_name="resnet18")
   trainer.train(train_loader, val_loader, epochs=5)
   
   # Standard conformal prediction
   label_predictor = LABEL(model=model, alpha=0.1)  # 90% coverage target
   label_predictor.calibrate(cal_dataset=cal)
   
   # Covariate shift adaptive conformal prediction
   cal_embeddings = extract_embeddings(model, cal, batch_size=32)
   test_embeddings = extract_embeddings(model, test, batch_size=32)
   
   cov_predictor = CovariateLabel(model=model, alpha=0.1)
   cov_predictor.calibrate(
       cal_dataset=cal,
       cal_embeddings=cal_embeddings,
       test_embeddings=test_embeddings
   )
   
   # Evaluate: CovariateLabel adapts to distribution shift
   # and provides more robust coverage guarantees

The key difference: ``CovariateLabel`` fits KDEs to detect and correct for covariate shifts between your calibration and test distributions. This gives you more reliable coverage guarantees when patient populations change.

**Note:** This is a preliminary example showing the direction we're heading. The core calibration framework is already in place, and we're actively expanding it with more techniques for non-I.I.D. settings. Expect more comprehensive examples and documentation as we continue building out these capabilities. This is an area we're really excited about because we think it can make a real difference in getting models deployed responsibly.


----


Bounty System: Your Path to the Research Initiative
----------------------------------------------------

`View Bounty List → <https://docs.google.com/spreadsheets/d/1PNMgDe-llOm1SM5ZyGLkmPysjC4wwaVblPLAHLxejTw/edit#gid=159213380>`_

When we redesigned PyHealth's approach to data loading and task processing for better efficiency, a lot of the old models broke compatibility with the new API. Rather than let these models languish, we set up a bounty system to get them working again and to expand our dataset coverage.

The response has been great! High-quality contributions (implementing models, adding datasets, or creating new task definitions) serve as your entry point to our Research Initiative. The "reward" isn't monetary—it's access to mentorship and our network of industry and academic collaborators. Make a solid contribution through the bounty system, and you can get matched with expert mentors to work on real research. It's been a win-win: we get more models and datasets available, and contributors get hands-on experience with healthcare AI plus a direct path into research collaboration.

**Recent additions we're excited about:**

**New Datasets:**

- **ChestXray14**: Large-scale chest X-ray dataset with 14 disease categories
- **DreamT**: Sleep staging dataset for polysomnography analysis
- **BMS-HS**: Heart sound recording dataset

**Updated Models:**

- Several classic EHR models now working with PyHealth 2.0
- Expanded task coverage across different datasets

**Want to get involved?**

1. Browse the `bounty list <https://docs.google.com/spreadsheets/d/1PNMgDe-llOm1SM5ZyGLkmPysjC4wwaVblPLAHLxejTw/edit#gid=159213380>`_ and pick something that interests you
2. Follow our `contribution guide <https://pyhealth.readthedocs.io/en/latest/how_to_contribute.html>`_
3. Submit a PR

We're always happy to help new contributors get started—just reach out on Discord if you have questions.


----


PyHealth Research Initiative: From Bounties to Publications
-----------------------------------------------------------

`Learn More → <https://pyhealth.readthedocs.io/en/latest/research_initiative.html>`_

We're really proud to share that our PyHealth Research Initiative has published three papers at ML4H this year! This program is all about pairing contributors with expert mentors to work on real research projects from start to finish—going from an idea to a published paper.

**Here's how it connects to the bounty system:** High-quality bounty contributions (adding datasets, models, or tasks) serve as your application to the Research Initiative. Once you've demonstrated your skills and commitment through a solid PR, we match you with a mentor from our network of academic and industry collaborators. This is where the industry connections and collaboration opportunities come in—they're all part of the Research Initiative.

The initiative has been running for a while now, and we've built up a strong network working on diverse healthcare AI problems. If you're interested in doing open-source research with mentorship from experienced researchers, this is the path.

**What you get:**

- Direct mentorship from experts in healthcare AI research
- Hands-on experience with real research problems
- Access to our network of academic and industry collaborators
- A clear path from contribution to publication
- Industry connections and collaboration opportunities

**How to join:**

1. Join our Discord community: `https://discord.gg/mpb835EHaX <https://discord.gg/mpb835EHaX>`_
2. Make a high-quality contribution via the bounty system (add a new dataset, model, or task with comprehensive examples)
3. Get matched with a mentor based on your interests and contribution

We're actively recruiting new contributors. Whether you're a student looking to get into research, a practitioner interested in healthcare AI, or a researcher wanting to collaborate, there's room for you here.


----


Join Us for PyHealth Casual Chats
----------------------------------

We run informal "Casual Chats" sessions where you can ask questions, discuss research ideas, or just talk about what you're working on with PyHealth. These are completely open—no preparation needed, just drop in if you're free.

- **When**: Every other Friday, 12:00 PM - 1:00 PM CT
- **Where**: `Zoom link <https://illinois.zoom.us/j/83607767000?pwd=xXKdKKs2YBH8d0UMWeUiYNEl0lhDoU.1>`_
- **Format**: Open Q&A, feature discussions, research questions, or whatever's on your mind

`Add to your calendar → <https://calendar.google.com/calendar/event?action=TEMPLATE&tmeid=NGY5NXM1ZnF1MWQ1cnFpdmxxMzNodTNlM2pfMjAyNTEyMTJUMTgwMDAwWiBqb2hud3UzQGlsbGlub2lzLmVkdQ&tmsrc=johnwu3%40illinois.edu&scp=ALL>`_


----


Get Involved
------------

We're always looking for feedback, feature requests, and contributions. Here's how to stay connected:

- **Discord**: `https://discord.gg/mpb835EHaX <https://discord.gg/mpb835EHaX>`_ — Join the community, ask questions, share what you're working on
- **GitHub**: `https://github.com/sunlabuiuc/PyHealth <https://github.com/sunlabuiuc/PyHealth>`_ — Star the repo, open issues, submit PRs
- **Mailing List**: `Subscribe here <https://docs.google.com/forms/d/e/1FAIpQLSfpJB5tdkI7BccTCReoszV9cyyX2rF99SgznzwlOepi5v-xLw/viewform?usp=header>`_ — Get updates delivered to your inbox

Please don't hesitate to reach out! We love hearing what features you'd like to see in PyHealth and how you're using it in your work.

Thanks for being part of the community!

