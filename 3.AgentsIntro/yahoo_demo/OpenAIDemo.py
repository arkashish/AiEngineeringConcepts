## Using OpenRouter
# from dotenv import load_dotenv
# from openai import OpenAI
# import os

# load_dotenv()
# # print(os.getenv("OPEN_ROUTER_API_KEY"))
# client = OpenAI(
#     api_key=os.getenv("OPEN_ROUTER_API_KEY"),
#     base_url="https://openrouter.ai/api/v1",
# )

# response = client.chat.completions.create(
#     model="openai/gpt-4o",
#     messages=[
#         {
#             "role": "user",
#             "content": "What is Apple's stock ticker and current price?"
#         },
#     ],
#     max_tokens=500,

# )

# print(response.choices[0].message.content)

from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": "What is Apple's stock ticker?",
        }
    ],
    
    max_tokens=200,
)

print(response.choices[0].message.content)
