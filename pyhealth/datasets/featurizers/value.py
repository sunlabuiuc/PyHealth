from dataclasses import dataclass


@dataclass
class ValueFeaturizer:

    def encode(self, value):
        return value


if __name__ == "__main__":
    featurizer = ValueFeaturizer()
    print(featurizer)
    print(featurizer.encode(2))