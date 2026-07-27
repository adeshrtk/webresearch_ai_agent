import os
import json
import requests
import ollama
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path

SEARCH_URL = "https://ollama.com/api/web_search"
MODEL = "qwen3.5:4b"

# Search web using Ollama web search
def search_web(query):
    api_key = os.getenv("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError("OLLAMA_API_KEY is required for Ollama web search.")

    response = requests.post(
        SEARCH_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"query": query, "max_results": 5},
        timeout=30
    )

    if response.status_code == 401:
        raise RuntimeError("Unauthorized: Invalid or missing OLLAMA_API_KEY.")

    response.raise_for_status()
    return response.json().get("results", [])

# Fetch full web page content
def fetch_text(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()        
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return ""
    
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)

def main():
    user_prompt = input("Enter your research prompt: ").strip()
    if not user_prompt:
        print("Prompt cannot be empty.")
        return
    
    try:
        results = search_web(user_prompt)
    except RuntimeError as e:
        print(f"Error: {e}")
        return
    except requests.RequestException as e:
        print(f"Web search request failed: {e}")
        return

    # For each url in web search results, fetch full content
    pages = []
    for item in results:
        url = item.get("url")
        if not url:
            continue
        
        print(f"Fetching: {url}")
        page_text = fetch_text(url)
        
        pages.append({
            "title": item.get("title", ""),
            "url": url,
            "snippet": item.get("content", ""),
            "page_text": page_text
        })
        
    # Prompt to send to Qwen model with web data
    prompt = f"""
    User request: {user_prompt}
    
    Use these web results and page contents to answer in markdown format.json
    
    Data:
    {json.dumps(pages, ensure_ascii=False)}
    """
    
    # Invoke local Qwen model
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    
    digest = response['message']['content']
    
    # Build a unique filename using today's date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"digest_{timestamp}.md"
    
    # Save the digest to a markdown file
    with open(filename, "w") as f:
        f.write(digest)
        
    print(f"Digest saved to {filename}")
    
if __name__ == "__main__":
    main()