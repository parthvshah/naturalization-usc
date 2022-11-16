

for_test_raw = open("./data/for_test_raw.txt", "r")
test_in = open("./data/test_input.txt", "w+")

for line in for_test_raw.readlines():
    tokens = line.split()
    new_s = ""
    for token in tokens:
        if not(token[0] == "(" and token[-1] == ")"):
            new_s = new_s + " " + token
    test_in.write(new_s + "\n")