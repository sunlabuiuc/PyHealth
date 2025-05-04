import unittest
import torch
from pyhealth.models.chestxray_vgg import ChestXRayVGG16

class TestChestXRayVGG16(unittest.TestCase):
    def setUp(self):
        self.model = ChestXRayVGG16(n_classes=2, pretrained=False)
        self.input = torch.randn(4, 3, 224, 224)  # batch of 4 images

    def test_forward_output_shape(self):
        output = self.model(self.input)
        self.assertEqual(output.shape, (4, 2), "Output shape should be (batch_size, n_classes)")

    def test_requires_grad(self):
        # Check that parameters have gradients
        params = list(self.model.parameters())
        self.assertTrue(all(p.requires_grad for p in params if p.requires_grad is not None))

if __name__ == '__main__':
    unittest.main()
