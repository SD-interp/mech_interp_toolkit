from typing import cast

import torch
from transformers import PreTrainedTokenizer


class ChatTemplateTokenizer:
    """
    A wrapper for Hugging Face tokenizers that applies a chat template and tokenizes the input.

    This class simplifies the process of formatting and tokenizing prompts for chat-based models.
    It handles the application of a system prompt and user prompts, and then tokenizes the
    formatted text into a dictionary of tensors.

    Args:
        tokenizer (PreTrainedTokenizer): The Hugging Face tokenizer to wrap.
        suffix (str, optional): A suffix to append to the formatted prompt. Defaults to "".
        system_prompt (str, optional): The system prompt to use. Defaults to "You are a strategic planning assistant that follows user instructions carefully".
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        suffix: str = "",
        system_prompt: str = "",
    ):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.system_prompt: str = system_prompt
        self.structured_prompt: list[str] | None = None
        self.suffix = suffix

    def _apply_chat_template(
        self, base_prompts: str | list[str], thinking: bool = False
    ) -> list[str]:
        """
        Applies the chat template to the base prompts.

        Args:
            base_prompts (str | list[str]): The base prompt or a list of base prompts.
            thinking (bool, optional): Whether to enable the 'thinking' template. Defaults to False.

        Returns:
            list[str]: A list of formatted prompts.
        """
        if isinstance(base_prompts, str):
            base_prompts = [base_prompts]

        instruct_syntax_prompts = [
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            for user_prompt in base_prompts
        ]

        formatted_prompts = [
            self.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_special_tokens=True,
                add_generation_prompt=True,
                enable_thinking=thinking,
            )
            + self.suffix  # type: ignore
            for prompt in instruct_syntax_prompts
        ]
        formatted_prompts = cast(list[str], formatted_prompts)
        self.structured_prompt = formatted_prompts
        return formatted_prompts

    def _encode(self, formatted_prompts: list[str]) -> dict[str, torch.Tensor]:
        """
        Tokenizes the formatted prompts.

        Args:
            formatted_prompts (list[str]): A list of formatted prompts.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the tokenized input IDs and attention mask.
        """
        tokenized = self.tokenizer(
            formatted_prompts,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        )
        tokenized = cast(dict[str, torch.Tensor], tokenized)
        return tokenized

    def __call__(self, prompts: str | list[str], thinking: bool = False) -> dict[str, torch.Tensor]:
        """
        Applies the chat template and tokenizes the prompts.

        Args:
            prompts (str | list[str]): The prompt or a list of prompts.
            thinking (bool, optional): Whether to enable the 'thinking' template. Defaults to False.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the tokenized input IDs and attention mask.
        """
        formatted_prompts = self._apply_chat_template(prompts, thinking=thinking)
        tokenized = self._encode(formatted_prompts)
        return tokenized

    def __getattr__(self, name: str):
        """
        Delegate attribute and method lookups to the underlying tokenizer.

        Args:
            name (str): The name of the attribute or method.

        Returns:
            The attribute or method from the underlying tokenizer.

        Raises:
            AttributeError: If the attribute is not found in the underlying tokenizer.
        """
        return getattr(self.tokenizer, name)

    def __dir__(self):
        """
        Return a list of attributes including those from the underlying tokenizer.

        Returns:
            list[str]: A list of attribute names.
        """
        return list(set(list(super().__dir__()) + dir(self.tokenizer)))
