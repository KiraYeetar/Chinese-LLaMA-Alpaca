import argparse
import os
import sys

import fire
import gradio as gr
import torch
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer

generation_config = dict(
    temperature=0.2,
    top_k=40,
    top_p=0.9,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.3,
    max_new_tokens=400
)

def main(
    base_model: str = "THUDM/chatglm-6b",
    lora_model: str = None,
    tokenizer_path: str = None,
    share_gradio: bool = False,
):
    load_type = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if tokenizer_path is None:
        tokenizer_path = args.lora_model
        if lora_model is None:
            tokenizer_path = args.base_model
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)

    base_model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
    )

    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    tokenizer_vocab_size = len(tokenizer)
    print(f"Vocab of the base model: {model_vocab_size}")
    print(f"Vocab of the tokenizer: {tokenizer_vocab_size}")
    if model_vocab_size != tokenizer_vocab_size:
        assert tokenizer_vocab_size > model_vocab_size
        print("Resize model embeddings to fit tokenizer")
        base_model.resize_token_embeddings(tokenizer_vocab_size)
    if lora_model is not None:
        print("loading peft model")
        model = PeftModel.from_pretrained(base_model, lora_model, torch_dtype=load_type)
    else:
        model = base_model

    if device == torch.device('cpu'):
        model.float()

    model.to(device)
    model.eval()

    def evaluate(
        input_text=None,
    ):
        inputs = tokenizer(input_text, return_tensors="pt")  # add_special_tokens=False ?
        generation_output = model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs['attention_mask'].to(device),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            **generation_config
        )
        s = generation_output[0]
        output = tokenizer.decode(s, skip_special_tokens=True)
        print(output)
        response = output
        yield response

    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=2,
                label="Context",
                placeholder="今晚月色真美啊",
            ),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="Chinese-LLaMA-Alpaca (merged from chinese-alpaca)",
    ).queue().launch(server_name="0.0.0.0", share=share_gradio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--lora_model', default=None, type=str, help="If None, perform inference on the base model")
    parser.add_argument('--tokenizer_path', default=None, type=str)
    parser.add_argument('--share_gradio', default=False, type=bool)
    args = parser.parse_args()
    main(args.base_model, args.lora_model, args.tokenizer_path, share_gradio=args.share_gradio)
