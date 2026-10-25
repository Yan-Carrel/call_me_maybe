try:
    import json
    import sys
    import src.decoding
    from src.parsing import parsing
    from src.llm_sdk import Small_LLM_Model
    from src.decoding.llm import Llm
except KeyboardInterrupt as e:
    print(e)


if __name__ == "__main__":
    try:
        prompts, funcs_def, funcs_def_json_path, output_name, model_name = (
            parsing())
        small_llm = Small_LLM_Model(model_name=model_name)
        vocab_file_path = small_llm.get_path_to_vocab_file()
        with open(vocab_file_path, "r", encoding="utf-8") as file:
            vocabs = json.load(file)
        with open(funcs_def_json_path, "r", encoding="utf-8") as funct_file:
            funcs_def_json = json.load(funct_file)
        llm = Llm(
            llm=small_llm, vocabs=vocabs, vocabs_str=list(vocabs.keys()),
            model_name=model_name
            )
        if len(funcs_def) < 1:
            print("Error, functions_definition.json cannot be an empty.")
            sys.exit(0)
        decoder = src.decoding.Decoder(
            llm=llm, funcs_def_json=json.dumps(funcs_def_json),
            funcs_def=funcs_def, prompts=prompts,
            vocabs=vocabs, file_name=output_name
            )
        decoder.generation_loop()
    except (KeyboardInterrupt, Exception) as e:
        print(e)
