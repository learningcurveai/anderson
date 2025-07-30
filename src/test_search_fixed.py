#!/usr/bin/env python3

import requests
import json

# Test the search functionality with proper session initialization
def test_search():
    base_url = "http://localhost:8888"
    
    # Step 1: Visit the index page to initialize session
    print("Step 1: Visiting index page to initialize session...")
    response = requests.get(f"{base_url}/")
    print(f"Index page response: {response.status_code}")
    
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
            if len(result['search_images']) > 0:
                print("✅ SUCCESS: Images were extracted correctly!")
                for i, url in enumerate(result['search_images']):
                    print(f"  Image {i+1}: {url}")
            else:
                print("❌ PROBLEM: search_images array is empty")
        else:
            print("\n❌ NO search_images found in response")

if __name__ == "__main__":
    test_search()
