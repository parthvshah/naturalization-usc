import json

key_path = "./evalKey.json"


class evalManager:
    def __init__(self, soft_matching=False):
        self.key = dict()
        self.soft_matching = soft_matching

        with open(key_path, "r") as key_file:
            key_as_str = key_file.read()

        self.key = json.loads(key_as_str)

    def match(self, left, right):
        if self.soft_matching:
            left_group = self.group(left)
            right_group = self.group(right)

            return left_group == right_group

        # Hard matching
        else:
            return left == right

    # TODO: Handle failure
    def group(self, embedding):
        for g in self.key.keys():
            if embedding in self.key[g]:
                return g

        return "INVALID"


if __name__ == "__main__":
    manager = evalManager()
