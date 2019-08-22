import datetime

def cmd_to_dict(cmd, list_format=False):
    """ Gets a dictionary of relevant args from a command. """
    if list_format:
        modified_cmd = cmd
        cmd = " ".join(cmd)
    else:
        modified_cmd = cmd.split()[2:]

    if "&" in modified_cmd:
        modified_cmd.remove("&")


    argmap = {}
    key = None

    for c in modified_cmd:
        if c[0] == "-":
            key = c[2:]
            argmap[key] = "true"
        else:
            argmap[key] = c

    argmap["cmd"] = cmd

    if "title" not in argmap:
        argmap["title"] = dict_to_title(argmap)

    return argmap


def dict_to_title(argmap):
    """ Converts a map of the relevant args to a title. """

    # python 3, this will be sorted and deterministic.
    print(argmap)
    exclude_list = ["train-file", "classes-file", "val-file", "cmd"]
    return "_".join([k + "=" + v for k, v in argmap.items() if k not in exclude_list])


def cmd_to_title(cmd, list_format=False):
    """ Gets a title from a command. """
    argmap = cmd_to_dict(cmd, list_format)
    return argmap["title"] + "_" + formatted_date()


def formatted_date():
    date = datetime.datetime.now()
    y = str(date).split(" ")
    return y[0] + "_" + y[1].split(".")[0]