from pyhealth.models.smart_adv import SmartAdversary
import torch


def main():
    x = torch.randn(10, 1)
    model = SmartAdversary(n_sensitive=1)
    output = model(x)
    print("Output shape:", output.shape)
    print(output)

if __name__ == "__main__":
    main()


#PYTHONPATH=. python examples/test_smart_adversary.py
