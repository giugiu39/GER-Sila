import os
import json
from dotenv import load_dotenv
import anthropic
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from schemas import Message

load_dotenv()

_MODEL = os.getenv("AGENT_MODEL", "claude-sonnet-4-20250514")

_SYSTEM_PROMPT = (
    "You are an assistant for the workers of Parco Nazionale della Sila.\n"
    "You have access to a MySQL database with the following table:\n\n"
    "detections (id, node_name, num_persone, emotion, timestamp)\n"
    "- node_name: name of the Raspberry Pi device (e.g. \"raspi-01\"), known by the workers\n"
    "- num_persone: number of people detected in the frame\n"
    "- emotion: one of happiness, neutral, surprise, sadness, fear, disgust, contempt, anger (NULL if not yet classified)\n"
    "- timestamp: unix timestamp in seconds\n\n"
    "When you need data, write a valid MySQL SELECT query and call the execute_query tool.\n"
    "Only SELECT queries are allowed — never INSERT, UPDATE, DELETE, or DROP.\n"
    "If the execute_query tool returns \"Error: only SELECT queries are allowed\", it means the query was blocked — respond to the user politely that you are not authorized to perform that operation.\n"
    "If the user asks to insert, modify, delete data, or perform any non-query operation, respond politely that you are not authorized to perform that operation without calling the tool.\n"
    "Always respond in natural language. Be concise and direct.\n"
    "Never show your reasoning, intermediate thoughts, or what queries you are executing — only return the final answer to the user.\n"
    "Timestamps are unix seconds — use FROM_UNIXTIME() for time-based queries."
)

_TOOLS = [
    {
        "name": "execute_query",
        "description": "Execute a SQL SELECT query on the detections table and return the results as a list of rows.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A valid MySQL SELECT query on the detections table.",
                }
            },
            "required": ["query"],
        },
    }
]


async def _execute_query(query: str, db: AsyncSession) -> list[dict] | str:
    if not query.strip().upper().startswith("SELECT"):
        return "Error: only SELECT queries are allowed"
    try:
        result = await db.execute(text(query))
        rows = result.fetchall()
        columns = list(result.keys())
        return [dict(zip(columns, row)) for row in rows]
    except Exception:
        return []


async def run_agent(messages: list[Message], db: AsyncSession) -> str:
    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    api_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

    for _ in range(10):
        result = await client.messages.create(
            model=_MODEL,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            tools=_TOOLS,
            messages=api_messages,
        )

        tool_uses = [block for block in result.content if block.type == "tool_use"]

        if not tool_uses:
            for block in result.content:
                if block.type == "text":
                    return block.text
            return ""

        api_messages.append({"role": "assistant", "content": result.content})

        tool_results = []
        for tool_use in tool_uses:
            query = tool_use.input.get("query", "")
            output = await _execute_query(query, db)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": json.dumps(output, default=str),
                }
            )

        api_messages.append({"role": "user", "content": tool_results})

    return ""
