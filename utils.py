
def remove_punct(string):
    return " ".join("".join([ item for item in string if (item.isalpha() or item == " ") ]).split())



