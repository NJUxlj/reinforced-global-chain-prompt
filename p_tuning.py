from peft import PromptEncoder, PromptEncoderConfig

config = PromptEncoderConfig(
    peft_type="P_TUNING",
    task_type="SEQ_2_SEQ_LM",
    num_virtual_tokens=20,
    token_dim=768,
    num_transformer_submodules=1,
    num_attention_heads=12,
    num_layers=12,
    encoder_reparameterization_type="MLP",
    encoder_hidden_size=768,
)

prompt_encoder = PromptEncoder(config)