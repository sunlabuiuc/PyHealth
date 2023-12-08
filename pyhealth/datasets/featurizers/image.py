import PIL.Image
import torchvision.transforms as transforms


class ImageFeaturizer:

    def __init__(self):
        self.transform = transforms.Compose([transforms.ToTensor()])

    def encode(self, value):
        image = PIL.Image.open(value)
        image.load()  # to avoid "Too many open files" errors
        image = self.transform(image)
        return image


if __name__ == "__main__":
    sample_image = "/srv/local/data/COVID-19_Radiography_Dataset/Normal/images/Normal-6335.png"
    featurizer = ImageFeaturizer()
    print(featurizer)
    print(type(featurizer))
    print(featurizer.encode(sample_image))