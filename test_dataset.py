from pyhealth.datasets import CheXpertCXRDataset

def main():
  dataset = CheXpertCXRDataset(
      root=r'C:\Users\nanda\Downloads\CheXpert-v1.0-small',
      config_path=r'C:\Users\nanda\OneDrive\Documents\GitHub\PyHealth\pyhealth\datasets\configs\chexpert_cxr.yaml',
      dev=True,
  )
  dataset.stats()
  samples = dataset.set_task()
  print(samples[0])
  pneumonia_samples = [sample for sample in samples if sample['Pneumonia'] == 1]
  print("Number of pneumonia samples:", len(pneumonia_samples))
  print("Number of non-pneumonia samples:", len(samples) - len(pneumonia_samples))

if __name__ == '__main__':
    main()