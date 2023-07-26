from PIL import Image  
from pyhealth.models.medclip_zeroshot import * 

# STEP 1: load data
image = Image.open('/home/wuzijian1231/Datasets/MedCLIP/example_data/view1_frontal.jpg')
processor = MedCLIPProcessor()
inputs = processor(images=image, return_tensors="pt")

# STEP 2: define model
model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.from_pretrained()
clf = PromptClassifier(model, ensemble=True)
clf.cuda()

# STEP 3: define prompts
cls_prompts = process_class_prompts(generate_chexpert_class_prompts(n=10))
inputs['prompt_inputs'] = cls_prompts

# STEP 4: make classification
output = clf(**inputs)
print(output)