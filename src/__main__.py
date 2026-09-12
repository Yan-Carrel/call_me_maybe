import json
from .decoder import Decoder
from parsing import parsing
from src.llm_sdk import Small_LLM_Model


if __name__ == "__main__":
    prompts, funcs_defs = parsing()
    llm = Small_LLM_Model()
    vocab_file_path = llm.get_path_to_vocab_file()
    with open(vocab_file_path, "r") as file:
        vocabs = json.load(file)
    decoder = Decoder(llm, prompts, funcs_defs, vocabs)
    decoder.generation_loop()
