import pytest
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
from mech_interp_toolkit.tokenizer import ChatTemplateTokenizer

@pytest.fixture
def tokenizer() -> PreTrainedTokenizer:
    """Fixture for a pretrained tokenizer."""
    return AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")

@pytest.fixture
def chat_tokenizer(tokenizer: PreTrainedTokenizer) -> ChatTemplateTokenizer:
    """Fixture for a ChatTemplateTokenizer."""
    return ChatTemplateTokenizer(tokenizer)

def test_initialization(tokenizer: PreTrainedTokenizer):
    """Tests the initialization of the ChatTemplateTokenizer."""
    chat_tokenizer = ChatTemplateTokenizer(tokenizer, suffix="<|end|>", system_prompt="You are a helpful assistant.")
    assert chat_tokenizer.tokenizer == tokenizer
    assert chat_tokenizer.suffix == "<|end|>"
    assert chat_tokenizer.system_prompt == "You are a helpful assistant."
    assert tokenizer.pad_token == tokenizer.eos_token

def test_apply_chat_template_single_prompt(chat_tokenizer: ChatTemplateTokenizer):
    """Tests the _apply_chat_template method with a single prompt."""
    base_prompt = "Hello, world!"
    formatted_prompts = chat_tokenizer._apply_chat_template(base_prompt)
    assert isinstance(formatted_prompts, list)
    assert len(formatted_prompts) == 1
    assert "You are a strtegic planning assitant that follows user instructions carefully" in formatted_prompts[0]
    assert "Hello, world!" in formatted_prompts[0]

def test_apply_chat_template_multiple_prompts(chat_tokenizer: ChatTemplateTokenizer):
    """Tests the _apply_chat_template method with multiple prompts."""
    base_prompts = ["Hello, world!", "How are you?"]
    formatted_prompts = chat_tokenizer._apply_chat_template(base_prompts)
    assert isinstance(formatted_prompts, list)
    assert len(formatted_prompts) == 2
    assert "Hello, world!" in formatted_prompts[0]
    assert "How are you?" in formatted_prompts[1]

def test_encode(chat_tokenizer: ChatTemplateTokenizer):
    """Tests the _encode method."""
    formatted_prompts = ["<|system|>You are a helpful assistant.<|end|><|user|>Hello, world!<|end|><|assistant|>",]
    tokenized = chat_tokenizer._encode(formatted_prompts)
    assert "input_ids" in tokenized
    assert "attention_mask" in tokenized
    assert isinstance(tokenized["input_ids"], torch.Tensor)
    assert isinstance(tokenized["attention_mask"], torch.Tensor)
    assert tokenized["input_ids"].shape[0] == 1
    assert tokenized["attention_mask"].shape[0] == 1

def test_call_single_prompt(chat_tokenizer: ChatTemplateTokenizer):
    """Tests the __call__ method with a single prompt."""
    prompt = "Hello, world!"
    tokenized = chat_tokenizer(prompt)
    assert "input_ids" in tokenized
    assert "attention_mask" in tokenized
    assert isinstance(tokenized["input_ids"], torch.Tensor)
    assert isinstance(tokenized["attention_mask"], torch.Tensor)
    assert tokenized["input_ids"].shape[0] == 1
    assert tokenized["attention_mask"].shape[0] == 1

def test_call_multiple_prompts(chat_tokenizer: ChatTemplateTokenizer):
    """Tests the __call__ method with multiple prompts."""
    prompts = ["Hello, world!", "How are you?"]
    tokenized = chat_tokenizer(prompts)
    assert "input_ids" in tokenized
    assert "attention_mask" in tokenized
    assert isinstance(tokenized["input_ids"], torch.Tensor)
    assert isinstance(tokenized["attention_mask"], torch.Tensor)
    assert tokenized["input_ids"].shape[0] == 2
    assert tokenized["attention_mask"].shape[0] == 2

def test_call_with_thinking(chat_tokenizer: ChatTemplateTokenizer):
    """Tests the __call__ method with the 'thinking' parameter."""
    prompt = "Hello, world!"
    tokenized = chat_tokenizer(prompt, thinking=True)
    assert "input_ids" in tokenized
    assert "attention_mask" in tokenized
    
    # Check if the 'thinking' template is applied (this is a basic check)
    decoded_text = chat_tokenizer.tokenizer.decode(tokenized["input_ids"][0])
    assert "You are a strtegic planning assitant that follows user instructions carefully" in decoded_text
    assert "Hello, world!" in decoded_text
