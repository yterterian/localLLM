# Local LLM System

This project provides a local LLM system with intelligent routing and context expansion.

## API Usage: Handling Punctuation and Special Characters in Prompts

When sending prompts to the API, you may use punctuation (commas, apostrophes, quotes, etc.) freely in the "content" field. Here are best practices to ensure your JSON is valid and your prompt is interpreted correctly:

- **Use double quotes** for all JSON keys and string values.
- **Apostrophes (single quotes)** do not need escaping inside double-quoted JSON strings.
- **Double quotes inside content** must be escaped with a backslash (`\"`).
- **Do not break JSON structure** with unclosed quotes or brackets.

**Example (apostrophes and punctuation are safe):**
```json
{
  "model": "qwen2.5-coder:32b",
  "messages": [
    {
      "role": "system",
      "content": "You're a chess expert"
    },
    {
      "role": "user",
      "content": "Kramnik is a chess player, who did he beat to become world champion and what was his strategy to win the match? Also, write the PGN's for the games he won and what game number."
    }
  ],
  "stream": false,
  "temperature": 0,
  "max_tokens": 512
}
```

**Example (using double quotes inside content):**
```json
{
  "role": "user",
  "content": "The term \"fringe benefits tax\" refers to a specific Australian tax."
}
```

**Tips:**
- Always check your JSON for proper closing of quotes and brackets.
- Use a JSON validator if unsure.
- If constructing JSON in code, use your language's JSON library to avoid manual escaping errors.

> If you encounter a 422 error, double-check your JSON formatting and ensure all strings are properly closed.

> **Note:** Please update this README with project details, usage instructions, and documentation as needed.
