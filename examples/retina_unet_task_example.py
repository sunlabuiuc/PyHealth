import numpy as np
from pyhealth.tasks.retina_unet_detection import RetinaUNetDetectionTask

def main():
    task = RetinaUNetDetectionTask()

    # fake data
    image = np.random.rand(128, 128, 3)
    mask = np.zeros((128, 128))
    mask[30:60, 40:80] = 1  # fake object

    sample = {"image": image, "mask": mask}

    output = task.process_sample(sample)

    print("Boxes:", output["boxes"])
    print("Labels:", output["labels"])


if __name__ == "__main__":
    main()
