from transformers import LlamaTokenizer
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="THUDM/chatglm-6b")
    parser.add_argument("--inputs", type=str, default="你好丁丂七丄丅丆万丈三上下丌不与丏丐丑丒专且丕世丗丘丙业丛东丝丞丟丠両丢丣两严並丧" * 200)
    args = parser.parse_args()

    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    inputs = args.inputs
    tokens = tokenizer.tokenize(inputs)
    print()
    print(">" * 20, "原始文字长度为: %d" % len(inputs), "<" * 20)
    print(">" * 20, "token分词长度为: %d" % len(tokens), "<" * 20)
    print()
