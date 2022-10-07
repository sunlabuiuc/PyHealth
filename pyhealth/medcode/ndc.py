class NDC:
    def __init__(self):
        self.valid_mappings = ["ATC3"]
        return

    def load_map_to_atc3(self):
        from MedCode import CodeMapping
        self.tool = CodeMapping("NDC11", "ATC4")
        self.tool.load()

    def map_to_atc3(self, ndc):
        result = []
        if ndc in self.tool.NDC11_to_ATC4:
            result += self.tool.NDC11_to_ATC4[ndc]
        result = list(dict.fromkeys([item[:-1] for item in result]))
        return result


if __name__ == "__main__":
    ndc = NDC()
    ndc.load_map_to_atc3()
    print(ndc.map_to_atc3("00597005801"))
