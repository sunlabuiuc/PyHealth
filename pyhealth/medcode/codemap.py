import importlib

from pyhealth.medcode.base_code import BaseCode


class CodeMap:
    def __init__(self, source, target):
        self.source = source
        self.target = target

        source_module = importlib.import_module(f"pyhealth.medcode.{source.lower()}")
        source_class = getattr(source_module, source)
        self.source_class: BaseCode = source_class()
        if target not in self.source_class.valid_mappings:
            raise ValueError(f"Invalid mapping: {source} -> {target}")
        return

    def map(self, source_code):
        return self.source_class.map_to(source_code, self.target)


if __name__ == "__main__":
    codemap = CodeMap("NDC", "ATC")
    print(codemap.map("00597005801"))
