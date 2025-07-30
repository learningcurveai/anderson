#!/usr/bin/env python3

import requests
import json

# Test the search functionality directly
def test_search():
    base_url = "http://localhost:8888"
    
    # Step 1: Initialize
    print("Step 1: Initializing...")
    init_data = {
        "user": "demo",
        "text": "hello, how are you doing"
    }
    
    response = requests.post(f"{base_url}/initialize", json=init_data)
    print(f"Initialize response: {response.status_code}")
    
    # Step 2: Send search query (chat_think)
    print("\nStep 2: Sending search query...")
    chat_data = {
        "user": "demo", 
        "text": "search for images of cats",
        "image": ""
    }
    
    response = requests.post(f"{base_url}/chat_think", json=chat_data)
    print(f"Chat think response: {response.status_code}")
    if response.status_code == 200:
        print(f"Chat think JSON: {response.json()}")
    
    # Step 3: Get the actual response (think_reply)
    print("\nStep 3: Getting AI response...")
    reply_data = {
        "user": "demo",
        "text": "search for images of cats"
    }
    
    response = requests.post(f"{base_url}/think_reply", json=reply_data)
    print(f"Think reply response: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Think reply JSON: {json.dumps(result, indent=2)}")
        
        # Check for search_images
        if 'search_images' in result:
            print(f"\n✅ FOUND search_images: {result['search_images']}")
            print(f"✅ Number of images: {len(result['search_images'])}")
        else:
            print("\n❌ NO search_images found in response")

if __name__ == "__main__":
    test_search()
