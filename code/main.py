"""
call functions from Test_Eval_Model to present the final result
"""

import os
import argparse
import pprint
import Test_Eval_Model

DATA_TYPE = ["original", "ekman", "group"]

def main(cli_args):
    model_name, test_type = cli_args.model_name, cli_args.test_type
    data_type="original" # incase of no match
    if "bert" in model_name.split("-"):
        for t in DATA_TYPE:
            if t in model_name:
                data_type = t
                break
    elif "roberta" in model_name.split("-"):
        for t in DATA_TYPE:
            if t in model_name:
                data_type = t
                break
        if data_type!="original": # temp, TODO
            raise NotImplementedError("test of other database with roberta not implemented yet")
        model_name = "roberta" # to get correct model path
    else:
        data_type = "original" #TODO other models only support 27 types for now
    # load model
    md = Test_Eval_Model.Eval(model_name)

    if test_type == "acc":
        acc = md.get_acc(os.path.abspath('..')+ '\\code\\data\\'+ data_type +'\\dev.tsv')
        # print the result
        print("Acc: {}".format(acc))
    elif test_type == "metrics":
        metrics = md.get_metrics(os.path.abspath('..')+'\\code\\data\\'+ data_type +'\\dev.tsv')
        pprint.pprint(metrics)
    else:
        raise NotImplementedError("Test type not implemented yet.")


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument("model_name", type=str, help="model name")
    cli_parser.add_argument("test_type", type=str, help="test type: acc/metrics")

    cli_args = cli_parser.parse_args()

    main(cli_args)