import argparse
import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from call_function import available_functions, call_function
from prompts import system_prompt

load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)


def main():
    print("Hello from bootdev-ai-agent!")

    parser = argparse.ArgumentParser(description="Chatbot")
    parser.add_argument("user_prompt", type=str, help="User prompt")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    # Now we can access `args.user_prompt`

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": args.user_prompt},
    ]
    response = None
    for _ in range(20):
        response = client.chat.completions.create(
            model="openrouter/free", messages=messages, tools=available_functions
        )
        if args.verbose:
            print("User prompt: %s" % args.user_prompt)
            print("Prompt tokens: %d" % response.usage.prompt_tokens)
            print("Response tokens: %d" % response.usage.completion_tokens)
        message = response.choices[0].message
        messages.append(message)
        if message.tool_calls:
            for tool_call in message.tool_calls:
                function_args = json.loads(tool_call.function.arguments or "{}")
                print(f"Calling function: {tool_call.function.name}({function_args})")
                result_message = call_function(tool_call)
                messages.append(result_message)
                if not result_message.get("content"):
                    raise Exception("No content")
                if args.verbose:
                    print(f"-> {result_message['content']}")
        else:
            response = message.content
            break
    if not response:
        print("No response")
        sys.exit(1)
    print(response)


if __name__ == "__main__":
    main()
