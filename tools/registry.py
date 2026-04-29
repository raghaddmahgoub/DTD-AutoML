class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, name, fn):
        self.tools[name] = fn

    def get(self, name):
        return self.tools.get(name)

    def list_tools(self):
        return list(self.tools.keys())