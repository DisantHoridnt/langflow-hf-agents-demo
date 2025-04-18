"""
Simple test script for the Hugging Face API integration.
This script avoids complex dependencies that might conflict.
"""

import os
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceHub

# Load environment variables
load_dotenv()

def test_huggingface_api():
    """Test basic connectivity to the Hugging Face API."""
    print("Testing connection to Hugging Face API...")
    
    # Check for API token
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not api_token:
        print("❌ HUGGINGFACEHUB_API_TOKEN not found in environment variables")
        return False
    
    # Create a simple LLM instance
    try:
        llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            huggingfacehub_api_token=api_token,
            model_kwargs={"temperature": 0.7, "max_new_tokens": 256}
        )
        
        # Test with a simple query
        result = llm.invoke("What is the capital of France?")
        
        print(f"✅ Successfully connected to Hugging Face API")
        print(f"Response: {result}")
        return True
    
    except Exception as e:
        print(f"❌ Error connecting to Hugging Face API: {e}")
        return False

if __name__ == "__main__":
    print("=== Simple Hugging Face API Test ===")
    
    success = test_huggingface_api()
    
    if success:
        print("\n✅ All tests passed successfully!")
    else:
        print("\n❌ Tests failed. Please check error messages above.")
