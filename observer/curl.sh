curl -X POST "http://h200-bar-196-011:8000/v1/chat/completions" \
  -H "Accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "",
    "messages": [{"role":"user","content":"Please calculate the result of 1+1= using tools?"}],
    "tool_choice": "required",
    "stream": false,
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "calculate",
          "description": "Execute simple mathematical calculation",
          "parameters": {
            "type": "object",
            "properties": {
              "a": {
                "description": "First number",
                "type": "integer"
              },
              "b": {
                "description": "Second number",
                "type": "integer"
              },
              "operation": {
                "description": "Operation to execute",
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"]
              }
            },
            "required": ["a", "b", "operation"]
          }
        }
      }
    ]
  }'