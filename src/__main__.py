import json
from src.decoding import Decoder
from src.parsing import parsing
from src.llm_sdk import Small_LLM_Model


if __name__ == "__main__":
    prompts, funcs_def, funcs_def_json_path = parsing()
    llm = Small_LLM_Model()
    vocab_file_path = llm.get_path_to_vocab_file()
    with open(vocab_file_path, "r", encoding="utf-8") as file:
        vocabs = json.load(file)
    with open(funcs_def_json_path, "r", encoding="utf-8") as funct_file:
        funcs_def_json = json.load(funct_file)
    decoder = Decoder(llm, json.dumps(funcs_def_json), funcs_def, prompts, vocabs)
    decoder.generation_loop()
