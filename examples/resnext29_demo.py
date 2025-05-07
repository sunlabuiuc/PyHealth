def main():
    from pyhealth.models import ResNeXt29
    import torch

    class DummyDataset:
        def __init__(self):
            self.task = "classification"
            self.input_schema = {"image": (torch.Tensor, (3, 32, 32))}
            self.output_schema = {"label": (int, 1)}
            self.output_size = 20

    dataset = DummyDataset()
    model = ResNeXt29(dataset=dataset, cardinality=8, depth=29, bottleneck_width=64)

    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    print("✅ Forward output shape:", out.shape)

if __name__ == "__main__":
    main()
