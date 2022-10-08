import importlib


class CodeMap:
    def __init__(self, source, target):
        self.source = source
        self.target = target

        source_module = importlib.import_module(f"pyhealth.medcode.{source.lower()}")
        source_class = getattr(source_module, source)
        self.source_class = source_class()
        if not target in self.source_class.valid_mappings:
            raise ValueError(f"{target} is not a valid mapping for {source}")
        load_map = getattr(self.source_class, f"load_map_to_{target.lower()}")
        load_map()
        return

    def map(self, source_code):
        return getattr(self.source_class, f"map_to_{self.target.lower()}")(source_code)


if __name__ == "__main__":
    codemap = CodeMap("NDC", "ATC3")
    print(codemap.map("00597005801"))
