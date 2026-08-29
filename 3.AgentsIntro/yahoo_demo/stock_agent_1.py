import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import os
import json
from openai import OpenAI
from yahoo_finance_demo import get_stock_price
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)


# TOOL_DEFINITION = [
#     {
#         "type": "function",
#         "function": {
#             "name": "get_stock_price",
#             "description": "Fetch the current price of a stock given its ticker symbol",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "ticker_symbol": {
#                         "type": "string",
#                         "description": "The stock ticker symbol, e.g. AAPL for Apple Inc."
#                     }
#                 },
#                 "required": ["ticker_symbol"]
#             }
#         }
#     }
# ]

TOOL_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": (
                "Fetch the current stock price. "
                "The tool requires the stock ticker symbol only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker_symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol such as MSFT, AAPL, or NVDA."
                    }
                },
                "required": ["ticker_symbol"],
                "additionalProperties": False
            }
        }
    }
]

SYSTEM_PROMPT = """
You are a helpful stock market assistant.

When the user asks about a stock price, identify the correct ticker symbol
and call the get_stock_price tool.

The get_stock_price tool requires exactly one argument:
ticker_symbol

Do not invent price, currency, or other arguments when calling the tool.

After receiving the tool result, provide a clear and friendly answer to the user.
Do not expose private chain-of-thought reasoning.
"""


FEW_SHOT_EXAMPLES = [
    #Example 1: Single stock lookup
    {
        "role": "user",
        "content": "How much is Apple stock right now?"
    },
    {
        "role": "assistant",
        "content": (
            "The user is asking about Apple's stock price. Apple's ticker symbol is AAPL."
            "Let me fetch the current price of Apple stock."
        ),
        "tool_calls": [
            {
                "id": "tool_call_01",
                "type": "function",
                "function": {
                    "name": "get_stock_price",
                    "arguments": json.dumps({
                        "ticker_symbol": "AAPL"
                    })
                }
            }
        ]
    },
    {
        "role": "tool",
        "tool_call_id": "tool_call_01",
        "content": json.dumps({"ticker": "AAPL", "price": 200.3, "currency": "USD"}),
    },
    {
        "role": "assistant",
        "content": (
            "Apple (AAPL) is currently trading at $200.30 per share in USD."
            "Keep in mind this is the last traded price and may shift slightly by the time you check again."
        )
    },
    # Example 2: Multi stock comparison
    {
        "role": "user",
        "content": "Compare the stock prices of Google and Microsoft"
    },
    {
        "role": "assistant",
        "content": (
            "The user wants to compare two stocks. Google's parent company "
            "Alphabet trades under the ticker symbol GOOGL, and Microsoft trades "
            "under the ticker symbol MSFT."
            "Let me fetch the current prices of both stocks."
        ),
        "tool_calls": [
            {
                "id": "tool_call_02",
                "type": "function",
                "function": {
                    "name": "get_stock_price",
                    "arguments": json.dumps({
                        "ticker_symbol": "GOOGL"
                    })
                }
            },
            {
                "id": "tool_call_03",
                "type": "function",
                "function": {
                    "name": "get_stock_price",
                    "arguments": json.dumps({
                        "ticker_symbol": "MSFT"
                    })
                }
            }
        ]
    },
    {
        "role": "tool",
        "tool_call_id": "tool_call_02",
        "content": json.dumps({"ticker": "GOOGL", "price": 280.12, "currency": "USD"}),
    },
    {
        "role": "tool",
        "tool_call_id": "tool_call_03",
        "content": json.dumps({"ticker": "MSFT", "price": 235.77, "currency": "USD"}),
    },
    {
        "role": "assistant",
        "content": (
            "Here's the comparison:\n\n"
            "- Alphabet/Google (GOOGL) is trading at $280.12 per share in USD.\n"
            "- Microsoft (MSFT) is trading at $235.77 per share in USD.\n\n"
            "Microsoft's stock is currently cheaper than Google's."
        )
    }
    

]


AVAILABLE_TOOLS = {
    "get_stock_price": get_stock_price,
}

def run_agent(user_query: str) -> None:
    """Single turn agentic loop with tool calling"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT_EXAMPLES,
        {"role": "user", "content": user_query},
    ]

    print(f"\n{'='*60}")
    print(f"User: {user_query}")
    print(f"{'='*60}\n")

    while True:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            tools=TOOL_DEFINITION,
            max_tokens=500,
        )

        choice = response.choices[0]
        assistant_msg = choice.message

        messages.append(assistant_msg)

        if assistant_msg.content:
            print(f"Assistant: {assistant_msg.content}\n")

        # No tool call -> final response
        if not assistant_msg.tool_calls:
            break

        # Execute every requested tool
        for tool_call in assistant_msg.tool_calls:

            tool_name = tool_call.function.name

            if tool_name not in AVAILABLE_TOOLS:
                print(f"Unknown tool: {tool_name}")
                continue

            arguments = json.loads(tool_call.function.arguments)

            print(
                f"Calling {tool_name} with arguments: {arguments}"
            )

            tool_function = AVAILABLE_TOOLS[tool_name]

            tool_result = tool_function(**arguments)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            })


def main():
    print("Stock price agent (type 'quit' to exit) ")
    print("-" * 45)

    while True:
        query = input("\nYou: ").strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        run_agent(query)

if __name__ == "__main__":
    main()