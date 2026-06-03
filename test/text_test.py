import torch


@torch.no_grad()
def check_cross_attention(model, enc_tokenizer, dec_tokenizer, device):
    model.eval()

    # two prefixes that should lead to very different continuations
    prompt_a = "The dragon soared over the burning castle as knights scattered below."
    prompt_b = "The marine biologist logged the coral bleaching data before the storm."

    enc_a = enc_tokenizer(prompt_a, return_tensors="pt", truncation=True, max_length=512).to(device)
    enc_b = enc_tokenizer(prompt_b, return_tensors="pt", truncation=True, max_length=512).to(device)

    # IDENTICAL decoder input for both
    dec_in = torch.tensor([[dec_tokenizer.bos_token_id]], device=device)

    out_a = model(input_ids=enc_a.input_ids, attention_mask=enc_a.attention_mask, decoder_input_ids=dec_in)
    out_b = model(input_ids=enc_b.input_ids, attention_mask=enc_b.attention_mask, decoder_input_ids=dec_in)

    diff  = (out_a.logits - out_b.logits).abs().mean().item()
    top_a = dec_tokenizer.decode([out_a.logits[0, -1].argmax().item()])
    top_b = dec_tokenizer.decode([out_b.logits[0, -1].argmax().item()])

    print(f"Mean |logit difference| (same decoder input, different prefix): {diff:.5f}")
    print(f"Predicted next token  |  A -> '{top_a}'   B -> '{top_b}'")
    if diff < 1e-4:
        print("\u26a0\ufe0f  Logits nearly identical \u2014 decoder is IGNORING RoBERTa (cross-attention dead).")
    else:
        print("\u2713  Decoder output depends on the prefix \u2014 cross-attention is active.")

    model.train()