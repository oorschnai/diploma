import click
import json
from keyword_parser import KeywordParser
import time
import psutil
import gc
import os
from synonyms_checker import SynonymsChecker


@click.command()
@click.option("--filename", "-FN", required=True, help="file with rpd name")
@click.option(
    "--save_to_file", "-S", required=False, help="name of output file (.json)"
)
@click.option(
    "--w2v_pretrained_model",
    "-PM",
    required=False,
    help="name of file with pretrained model (.model)",
)
@click.option(
    "--w2v_train_data",
    "-TD",
    required=False,
    help="name of file with train data (.json)",
)
def get_kw(filename, save_to_file, w2v_pretrained_model, w2v_train_data):
    start = time.time()
    f = open(filename)
    data = json.load(f, strict=False)
    f.close()

    start_index = 0
    end_index = 2

    if w2v_pretrained_model and w2v_train_data:
        synonyms_search = SynonymsChecker(
            model=w2v_pretrained_model, train_data=w2v_train_data
        )
    elif w2v_train_data:
        synonyms_search = SynonymsChecker(train_data=w2v_train_data)
    elif w2v_pretrained_model:
        synonyms_search = SynonymsChecker(model=w2v_pretrained_model)
    else:
        print("choose model file or data to train w2v")
        return

    parser = KeywordParser(synonyms_search)

    keywords = []

    for index in range(start_index, end_index):
        kw = parser.get_keywords(data[index : index + 1])
        keywords.append(kw[0])

        if save_to_file:
            for kws in keywords:
                for ent in kws["wiki"]["ner"]:
                    ent["score"] = str(ent["score"])
                    ent["index"] = str(ent["index"])
                    ent["start"] = str(ent["start"])
                    ent["end"] = str(ent["end"])
            with open(save_to_file, "w") as outfile:
                outfile.write(
                    json.dumps(
                        keywords,
                        ensure_ascii=False,
                        indent=2,
                    )
                )

        gc.collect()

    # clustering
    print("clustering all")
    keywords = {
        "keywords": keywords,
        "clustering": synonyms_search.get_synonyms(keywords),
    }

    end = time.time() - start
    print("Time:", end, "sec")
    print("CPU %", psutil.cpu_percent())
    print(
        "% available mem",
        psutil.virtual_memory().available * 100 / psutil.virtual_memory().total,
    )

    if save_to_file:
        with open(save_to_file, "w") as outfile:
            outfile.write(json.dumps(keywords, ensure_ascii=False, indent=4))

    return keywords


if __name__ == "__main__":
    get_kw()
