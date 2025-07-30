import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

import requests

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
    
    print("Image response keys:", image_data.keys())
    print("Images results:", image_data.get("images_results", [])[:3])
    
    # Process image results
    image_urls = []
    for item in image_data.get("images_results", [])[:3]:
        image_url = item.get("original", "")
        if image_url:
            image_urls.append(image_url)
    
    print("Extracted image URLs:", image_urls)
    
    # Include image URLs in the text result for conversation processing
    if image_urls:
        result += f" [SEARCH_IMAGES: {','.join(image_urls)}]"
    
    return result.strip()

# Test the search function
if __name__ == "__main__":
    result = search_engine("images of cats")
    print("\nFinal result:")
    print(result)
