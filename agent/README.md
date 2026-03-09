## Agent mode (tool calling)

When `AGENT_ENABLED=true`, the assistant runs in **agent mode**: the LLM returns a **single JSON object** per reply, separating:

- `say`: text to speak to the user
- `actions`: tool/function calls to execute

The system prompt is automatically appended on every LLM call with:

- The required JSON contract
- A list of available tools (built from tool docstrings)

### Response schema

The LLM must output exactly:

```json
{
  "say": "User-facing text to speak",
  "actions": [
    { "tool": "tool_name", "args": { "arg1": "value" } }
  ],
  "meta": { "optional": true }
}
```

If no tools are needed:

```json
{ "say": "…", "actions": [] }
```

### How the loop works

1. LLM returns JSON (`say` + `actions`)
2. `say` is spoken via Piper TTS
3. If `actions` is non-empty, tools run and their outputs are fed back into the LLM
4. Repeat until `actions` is empty (or iteration limit reached)

### Env configuration

- `AGENT_ENABLED` (default `true`)
- `AGENT_MAX_ITERATIONS` (default `5`)
- `AGENT_SPEAK_INTERMEDIATE` (default `true`)

