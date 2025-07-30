
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

# -----------------------
# Tools
# -----------------------

import base64
import requests
def analyze_image(base64_image_data):
    api_key = os.environ.get("OPENAI_API_KEY")
    
    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}"
    }

    # Use base64 data directly - no file I/O needed
    # Assume PNG format since that's what canvas.toDataURL() produces
    payload = {
      "model": "gpt-4o",
      "max_tokens": 1024,
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "Describe this image in detail."
            },
            {
              "type": "image_url",
              "image_url": {"url": f"data:image/png;base64,{base64_image_data}"}
            }
          ]
        }
      ],
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    # {'id': 'chatcmpl-9RRYu5W77ZoNKUUsdp84XFk2Pbxvg', 'object': 'chat.completion', 'created': 1716327832, 'model': 'gpt-4o', 
    #  'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The image displays multiple DVD or poster covers for ...'}, 
    #               'logprobs': None, 'finish_reason': 'stop'}], 
    #  'usage': {'prompt_tokens': 1118, 'completion_tokens': 142, 'total_tokens': 1260}, 'system_fingerprint': 'fp_927397958d'}
    print(response.json())
    result = response.json()['choices'][0]['message']['content']
    return result

#analyze_image(image_path="../input_images/image.demo.0.png")


def search_engine( query ):
    result = ""
    print("Search Engine Query:", query)

    # Serp API for organic results
    params = {
        "q": query,
        "api_key": os.environ["SERPAPI_API_KEY"],
        "engine": "google",
        "num": 3
    }
    
    response = requests.get("https://serpapi.com/search", params=params)
    data = response.json()
    
    # Process organic results
    for item in data.get("organic_results", [])[:3]:
        snippet = item.get("snippet", "")
        if snippet:
            result += snippet + " "
    
    # Get image results
    image_params = {
        "q": query,
        "api_key": os.environ["SERPAPI_API_KEY"],
        "engine": "google",
        "tbm": "isch",
        "num": 3
    }
    
    image_response = requests.get("https://serpapi.com/search", params=image_params)
    image_data = image_response.json()
    
    # Process image results
    image_urls = []
    for item in image_data.get("images_results", [])[:3]:
        image_url = item.get("original", "")
        if image_url:
            image_urls.append(image_url)
    
    # Include image URLs in the text result for conversation processing
    if image_urls:
        result += f" [SEARCH_IMAGES: {','.join(image_urls)}]"
    
    return result.strip()

#search_engine("What is James Bond's nationality ?")


# -----------------------
# Tool Calling Support
# -----------------------

import inspect
class Function:
    def __init__(self, func):
        self.func = func
        self.name = func.__name__
        self.arg_def = {}
        for name, param in inspect.signature(self.func).parameters.items():
            # {'ticker': {'type': <class 'str'>, 'value': None, 'default': <class 'inspect._empty'>}}
            self.arg_def[ name ]={"type":param.annotation, "value":None, "default":param.default}            
        # i.e. <class 'float'>
        self.return_type = inspect.signature(self.func).return_annotation
        
class ToolCall:
    def __init__(self, func):
        self.function = Function(func)
        self.arguments = {}

    def __call__(self, parameter):
        for key, arg_def in self.function.arg_def.items():
            self.function.arg_def[ key ]["value"]=parameter
            self.arguments[ key ]=parameter
        result = self.function.func(parameter)
        print("calling:", self.function.name, "args:", self.function.arg_def, "result:", result)
        return result

#tool = ToolCall(analyze_image)
#print(tool("../input_images/image.demo.0.png"))

#tool = ToolCall(search_engine)
#print(tool("What nationality is James Bond ?"))


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

    result = None
    for tool in tools:
        # i.e. {"tool": ToolCall(), "description":tool_description}
        if result_dict["tool"]==tool["tool"].function.name:    # ToolCall.function
            result = tool["tool"]( result_dict["parameter"] )  # ToolCall.__call__
        
    return result


_dialog = """
i'm fine, how are you doing
"""
# UNIT TESTS:
# how are you doing
# can you check Tesla Stock
# can you describe this image '../input_images/image.demo.0.png'
# do you know what happened recently with Charlotte Flair ?


# Test tool invocation support
tools = []
# Removed analyze_image from global tools - now handled directly in think_reply()
tools.append( {"tool": ToolCall(search_engine), "description":"name: search_engine, parameter: search_query, returns: web search results for factual queries. Use for explicit information requests like 'search for X', 'find Y', 'what happened with Z'. Do NOT use for greetings, casual conversation, or personal questions like 'how are you', 'i'm fine', 'hello'."} )

check_tools_invocation(_dialog, tools)


import re
import time
from openai import OpenAI

system_prompt="""You are a personal companion having a dialog with a User. \
System prompts will be provided to help drive the conversation including \
description of images or answers to questions. 
Try to elicit some topic the user is interested in to discuss.
"""

def call_open_ai(messages):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    # https://platform.openai.com/docs/models/overview
    # https://platform.openai.com/account/billing/preferences
    # https://platform.openai.com/account/usage
    
    completion = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return completion.choices[0].message.content

class Conversation:
    def __init__(self, system_prompt=""):
        self.messages = []
        self.tools = []
        
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

        # Removed analyze_image from tools - now handled directly in think_reply()
        self.tools.append( {"tool": ToolCall(search_engine),   "description":"name: search_engine, parameter: search_query, returns: web search results for factual queries. Use for explicit information requests like 'search for X', 'find Y', 'what happened with Z'. Do NOT use for greetings, casual conversation, or personal questions like 'how are you', 'i'm fine', 'hello'."} )

    def system(self, system_prompt):
        self.messages.append({"role": "system", "content": system_prompt})

    def assistant(self, assistant_prompt):
        self.messages.append({"role":"assistant", "content":assistant_prompt})

    def user(self, user_prompt):
        self.messages.append({"role":"user", "content":user_prompt})

    def chat(self):
        tool_response = None
        if self.tools and self.messages:
            _dialog = []
            #for message in self.messages:
            #    _dialog.append(message['content'])
            _dialog.append(self.messages[-1]['content'])
            _dialog = "\n".join(_dialog)
            tool_response = check_tools_invocation(_dialog, self.tools)
            if tool_response:
                # Clean the tool response for ChatGPT by removing image URLs
                import re
                search_pattern = r'\[SEARCH_IMAGES: ([^\]]+)\]'
                cleaned_tool_response = re.sub(search_pattern, '', tool_response).strip()
                
                # Extract the original query from the dialog for better context
                query = _dialog.strip()
                
                # Provide clear, structured context to ChatGPT
                self.system( f"Search results for '{query}': {cleaned_tool_response}. Tell the user you found relevant results and engage them conversationally about the topic." )
            
        response = call_open_ai(messages=self.messages)
        self.messages.append({"role":"assistant", "content":response})
        
        # Return both the response and the raw tool response for image extraction
        return response, tool_response
        
    def generate(self, user_prompt):
        self.messages.append({"role": "user", "content":user_prompt})
        response = call_open_ai(messages=self.messages)
        self.messages.append({"role":"assistant", "content":response})
        
        return response

import os
import requests
import numpy as np

import random
import time
from datetime import datetime
import json

from flask import Flask, request, jsonify, render_template, send_file


glob = {}

glob["prompt"]={}
glob["message_counter"] = {}
glob["current_app_state"] = {}
glob["conversation_object"] = {}
glob["pending_image"] = {}
glob["audio_toggle"] = {}


import openai

def generate_audio_only(user, text, stay_silent_and_blink=False):
    print("Generating audio with OpenAI TTS")

    if stay_silent_and_blink:
        # Use existing silence file
        audio_path = "static/silence.mp3"
    else:
        # === TEXT TO SPEECH via OpenAI ===
        openai.api_key = os.getenv("OPENAI_API_KEY")

        response = openai.audio.speech.create(
            model="tts-1",
            voice="onyx",  # Male voice
            #voice="nova",  # Female voice
            input=text
        )

        # User-specific alternating files to avoid browser caching
        if glob["audio_toggle"][user] == "A":
            audio_path = f"static/{user}_audio_a.mp3"
            glob["audio_toggle"][user] = "B"  # Next time use B
        else:
            audio_path = f"static/{user}_audio_b.mp3"
            glob["audio_toggle"][user] = "A"  # Next time use A

        with open(audio_path, "wb") as f:
            f.write(response.content)

    return audio_path



app = Flask(__name__)

@app.route("/")
def index():
    user = "demo"
    
    glob["message_counter"][ user ]=0
    glob["current_app_state"][ user ]="initializing"
    glob["conversation_object"][ user ]=Conversation(system_prompt=system_prompt)
    glob["audio_toggle"][ user ]="A"  # Start with A for each new user
                
    glob["prompt"][ user ]=""

    first_message = ""
    random.seed(time.time())
    first_message = random.choice( ["hey, hows it going", "hi, how are you", "hi, hows it going", "hey, how are you doing", "hey, hows it going", "hi, how are you", "hi, hows it going", "hello, how are you doing"] )

    print("The Initial Prompt is: ", system_prompt)    
    print("First Greeting: ", first_message)

    return render_template("widget.html", user=user, message=first_message)
    
@app.route("/initialize", methods=["POST"])
def initialize():
    user = request.get_json()["user"]
    text = request.get_json()["text"]
    image_prompt = response = ""
    
    print("Received: ", text, " from: ", user)
        
    if glob["current_app_state"][ user ]=="initializing":
        glob["current_app_state"][ user ]="running"

        # no call to chatGPT here but we will generate voice and wait for next chat
        glob["prompt"][ user ] += "Assistant: " + text + "\n"
        glob["conversation_object"][ user ].assistant( text )

        # generate audio after we get the assistant's next line
        audio_path = generate_audio_only( user, text )

    elif glob["current_app_state"][ user ]=="running":
        assert(False)

    # Return the actual filename from the audio generation
    audio_filename = audio_path.replace("static/", "")

    result = jsonify({"audio_file": audio_filename, "image_prompt": image_prompt, "response":response})
    return result
    
@app.route("/chat_think", methods=["POST"])
def chat_think():
    
    user = request.get_json()["user"]
    text = request.get_json()["text"]
    image_data = request.get_json()["image"]

    image_prompt = response = ""
    
    print("Received: ", text, " from: ", user)

    # Process any image that is sent with the chat
    if image_data != "":
        clean_base64 = image_data.replace('data:image/png;base64,', '')
        
        # Store image for processing during think_reply - no base64 in conversation!
        glob["pending_image"][user] = clean_base64
        
        # Add clean text to conversation (no massive base64)
        enhanced_text = text + ". [User has shared an image for analysis]"
        glob["prompt"][ user ] += "User: " + enhanced_text + "\n"
        glob["conversation_object"][ user ].user( enhanced_text )
        
        response = "pause and think"
    else:
        # Normal text processing
        glob["prompt"][ user ] += "User: " + text + "\n"
        glob["conversation_object"][ user ].user( text )

    # (A) Send Dialog to Observers ...
    
    dialog = glob["prompt"][ user ] 
    message_content = {}
    message_content["user"]=user
    message_content["dialog"]=dialog
    message_content["message_counter"]=glob["message_counter"][user]


    # Process fastpath queries
    stay_silent_and_blink = False
    if response=="pause and think":
        response = random.choice(["ok, let me think", "ok, let me see", "let me think, give me a sec", "give me a sec, let me think"])
        
    else:
        stay_silent_and_blink = True
        response = "thinking"
    
    start_time = time.time()
    
    # generate audio after we get the assistant's next line
    audio_path = generate_audio_only( user, response, stay_silent_and_blink=stay_silent_and_blink )

    end_time = time.time()
    print("* generate_audio_only Elapsed time:", end_time - start_time)    

    # Return the actual filename from the audio generation
    audio_filename = audio_path.replace("static/", "")

    result = jsonify({"audio_file": audio_filename, "image_prompt": image_prompt, "response":response})
    return result


@app.route("/think_reply", methods=["POST"])
def think_reply():
    
    user = request.get_json()["user"]
    text = request.get_json()["text"]
    
    image_prompt = response = ""

    glob["prompt"][ user ] += "Assistant: " 
    
    # Process any pending image during "thinking" time
    if user in glob["pending_image"]:
        base64_data = glob["pending_image"][user]
        
        print("Processing pending image during think_reply...")
        
        # Process image (takes time, but during expected thinking delay)
        image_description = analyze_image(base64_data)
        
        # Add description to conversation context
        glob["conversation_object"][user].system(f"Image Analysis: {image_description}. Please respond to the user about this image.")
        
        # Clean up stored image
        del glob["pending_image"][user]
        
        print("Image processing complete, description added to conversation.")
        
    # We call chatGPT for the next response. 
    start_time = time.time()

    # Continue the conversation ...
    response, tool_response = glob["conversation_object"][ user ].chat()
    
    glob["prompt"][ user ] += response + "\n"
    
    end_time = time.time()
    print("* call_open_ai Elapsed time:", end_time - start_time)
    
    # Extract search images from the RAW tool response (before ChatGPT processing)
    search_images = []
    import re
    search_pattern = r'\[SEARCH_IMAGES: ([^\]]+)\]'
    
    # Check tool_response first (raw search results)
    if tool_response:
        print(f"DEBUG: Checking tool_response for images: {tool_response}")
        match = re.search(search_pattern, tool_response)
        if match:
            image_urls_str = match.group(1)
            search_images = [url.strip() for url in image_urls_str.split(',') if url.strip()]
            print(f"DEBUG: Extracted {len(search_images)} search images from tool_response")
    
    # Fallback: check the final response (shouldn't be needed now)
    if not search_images:
        print(f"DEBUG: No images in tool_response, checking final response: {response}")
        match = re.search(search_pattern, response)
        if match:
            image_urls_str = match.group(1)
            search_images = [url.strip() for url in image_urls_str.split(',') if url.strip()]
            # Remove the search images pattern from the displayed response
            response = re.sub(search_pattern, '', response).strip()
            print(f"DEBUG: Extracted {len(search_images)} search images from final response")

    # Generate audio after we get the assistant's next line
    start_time = time.time()
    
    audio_path = generate_audio_only( user, response )

    end_time = time.time()
    print("* generate_audio_only Elapsed time:", end_time - start_time)


    # Return the actual filename from the audio generation
    audio_filename = audio_path.replace("static/", "")

    print(f"DEBUG: Sending response with search_images: {search_images}")
    print(f"DEBUG: search_images type: {type(search_images)}")
    print(f"DEBUG: search_images length: {len(search_images)}")
    
    result = jsonify({"audio_file": audio_filename, "image_prompt": image_prompt, "response":response, "search_images": search_images})

    return result



from http.server import BaseHTTPRequestHandler
class RequestHandler(BaseHTTPRequestHandler):
  def setup(self):
    BaseHTTPRequestHandler.setup(self)
    self.request.settimeout(120)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8888, threaded=True)
