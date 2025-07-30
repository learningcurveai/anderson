import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

from openai import OpenAI
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

def check_tools_invocation(dialog, tools):    
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    _dialog = dialog
    _tools = "Tools:\n"
    for tool in tools:
        _tools += "- "+tool["description"]+"\n"

    _prompt = """
    ### Instruction: If there is an explicit request for information in the input dialog, see if an input tool can answer it, then identify the tool and parameter needed, otherwise return '' for tool and parameter.

    ### Input:
    Input Dialog:
    %s
    
    Input Tools: 
    %s

    Format Instructions:
    %s
    
    ### Response:
    """

    tool_name_schema = ResponseSchema(name="tool", description="output the name of the tool")
    parameter_schema = ResponseSchema(name="parameter", description="output the value of parameter to be passed into the tool")

    response_schemas = [tool_name_schema, parameter_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    
    _format_instructions = output_parser.get_format_instructions()
    _prompt = _prompt % (_dialog, _tools, _format_instructions)
    
    messages=[{"role": "system", "content": ""}, {"role": "user", "content": _prompt}]
    completion = client.chat.completions.create(model="gpt-4o-mini", messages=messages) 
    result_dict = output_parser.parse(completion.choices[0].message.content)
    print("Tool Matching Result:", result_dict)
    return result_dict

# Test the improved tool description
if __name__ == "__main__":
    # Mock tool for testing
    class MockTool:
        def __init__(self, name):
            self.function = type('obj', (object,), {'name': name})
    
    # Updated tool description with negative examples
    tools = [{
        "tool": MockTool("search_engine"), 
        "description": "name: search_engine, parameter: search_query, returns: web search results for factual queries. Use for explicit information requests like 'search for X', 'find Y', 'what happened with Z'. Do NOT use for greetings, casual conversation, or personal questions like 'how are you', 'i'm fine', 'hello'."
    }]
    
    # Test queries that should NOT trigger the search tool
    casual_queries = [
        "hello how are you",
        "i'm fine, how are you doing", 
        "hey there",
        "good morning",
        "thanks for that",
        "okay cool",
        "nice to meet you"
    ]
    
    # Test queries that SHOULD trigger the search tool
    search_queries = [
        "search for images of cats",
        "find information about Tesla stock",
        "what happened with Charlotte Flair recently",
        "look up the weather in New York",
        "find pictures of mountains",
        "search for news about AI"
    ]
    
    print("=== Testing Casual Conversation (Should NOT trigger search) ===")
    for query in casual_queries:
        print(f"\nTesting: '{query}'")
        result = check_tools_invocation(query, tools)
        should_search = result.get('tool') == 'search_engine' and result.get('parameter', '').strip() != ''
        status = "❌ INCORRECTLY triggered search" if should_search else "✅ Correctly ignored"
        print(f"Result: {status}")
    
    print("\n\n=== Testing Search Requests (SHOULD trigger search) ===")
    for query in search_queries:
        print(f"\nTesting: '{query}'")
        result = check_tools_invocation(query, tools)
        should_search = result.get('tool') == 'search_engine' and result.get('parameter', '').strip() != ''
        status = "✅ Correctly triggered search" if should_search else "❌ INCORRECTLY ignored"
        print(f"Result: {status}")
        if should_search:
            print(f"Search parameter: '{result.get('parameter')}'")
