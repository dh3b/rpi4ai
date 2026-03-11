## Agent mode (tool calling)

When `AGENT_ENABLED=true`, the assistant runs in **agent mode** powered by
`llama-cpp-agent`’s `FunctionCallingAgent`:

- **User input** goes to a `FunctionCallingAgent` backed by the existing
  `llama-cpp-python` model (via `LlamaCppPythonProvider`).
- **Tools** are discovered from the existing `ToolRegistry` and wrapped as
  `LlamaCppFunctionTool` instances, so their signatures and docstrings become
  structured tool definitions for the model.
- **Tool calling & iteration** are handled internally by `llama-cpp-agent`,
  including guided sampling / grammars, JSON/tool schemas, and optional
  parallel function execution.

### Behaviour

- The agent can call one or more tools, possibly in parallel, and uses their
  results to synthesize a final natural-language reply.
- Intermediate user-facing messages from the agent are streamed through the
  same TTS + speaker pipeline when `AGENT_SPEAK_INTERMEDIATE=true`.
- The public API is `AgentController.run_turn(text, tts, speaker)`,
  returning the final string that was spoken.

### Env configuration

- `AGENT_ENABLED` (default `true`)
- `AGENT_SPEAK_INTERMEDIATE` (default `true`)

