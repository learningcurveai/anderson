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
    # i.e. result_dict={'tool': 'get_stock_price', 'parameter': 'NVIDIA'}
    print("Tool Matching Result:", result_dict)
    return result_dict

# Test tool matching
if __name__ == "__main__":
    # Mock tool for testing
    class MockTool:
        def __init__(self, name):
            self.function = type('obj', (object,), {'name': name})
    
    tools = [{"tool": MockTool("search_engine"), "description": "name: search_engine, parameter: request, returns: answer to request"}]
    
    # Test different queries
    test_queries = [
        "search for images of cats",
        "find pictures of dogs", 
        "show me images of mountains",
        "hello how are you",
        "what's the weather like"
    ]
    
    for query in test_queries:
        print(f"\nTesting query: '{query}'")
        result = check_tools_invocation(query, tools)
        print(f"Result: {result}")
