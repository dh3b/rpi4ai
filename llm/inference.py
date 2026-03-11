import logging
from typing import Iterator, Optional
from llama_cpp import Llama, LlamaGrammar
from config import LLMConfig

logger = logging.getLogger(__name__)

class LLMInference:
    def __init__(self, config: LLMConfig):
        self.config  = config
        self.model   = self._load_model()
        self.grammar = LlamaGrammar.from_file("/grammar/response.gbnf")
        self.history: list[dict[str, str]] = []

    def _load_model(self) -> Llama:
        logger.info(
            "Loading LLM  path=%s  n_ctx=%d  threads=%d",
            self.config.model_path,
            self.config.n_ctx,
            self.config.n_threads,
        )
        model = Llama(
            model_path=self.config.model_path,
            n_ctx=self.config.n_ctx,
            n_threads=self.config.n_threads,
            verbose=False,
        )
        logger.info("LLM ready")
        return model

    def _build_messages(self, extra_system_prompt: Optional[str] = None) -> list[dict[str, str]]:
        system_prompt = self.config.system_prompt
        if extra_system_prompt:
            system_prompt = f"{system_prompt}\n\n{extra_system_prompt}"
        return [
            {"role": "system", "content": system_prompt},
            *self.history,
        ]

    def chat(self, user_message: str, *, extra_system_prompt: Optional[str] = None) -> str:
        """
        Send user_message, append to history, return the full reply string.
        Blocks until generation is complete.
        """
        self.history.append({"role": "user", "content": user_message})
        logger.info("LLM ← '%s'", user_message)

        response = self.model.create_chat_completion(
            messages=self._build_messages(extra_system_prompt=extra_system_prompt),
            grammar=self.grammar,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        reply = response["choices"][0]["message"]["content"].strip()
        self.history.append({"role": "assistant", "content": reply})
        logger.info("LLM → '%s'", reply)
        return reply

    def stream_chat(self, user_message: str, *, extra_system_prompt: Optional[str] = None) -> Iterator[str]:
        """
        Stream the reply token-by-token.
        Yields each token as a string; full reply is saved to history
        after the stream is exhausted.

        Usage:
            full = ""
            for token in llm.stream_chat("Hello"):
                print(token, end="", flush=True)
                full += token
        """
        self.history.append({"role": "user", "content": user_message})
        logger.info("LLM (stream) ← '%s'", user_message)

        full_reply: list[str] = []

        for chunk in self.model.create_chat_completion(
            messages=self._build_messages(extra_system_prompt=extra_system_prompt),
            grammar=self.grammar,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            stream=True,
        ):
            token = chunk["choices"][0]["delta"].get("content", "")
            if token:
                full_reply.append(token)
                yield token

        reply = "".join(full_reply).strip()
        self.history.append({"role": "assistant", "content": reply})
        logger.info("LLM → '%s'", reply)

    def reset_history(self) -> None:
        """Clear conversation history"""
        self.history.clear()
        logger.info("Conversation history cleared")
