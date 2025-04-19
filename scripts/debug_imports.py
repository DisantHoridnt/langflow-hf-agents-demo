#!/usr/bin/env python
"""
Diagnostic script to investigate Langflow import issues.
"""

import sys
import traceback

def debug_imports():
    print("Python version:", sys.version)
    print("Python path:", sys.path)
    
    print("\nTrying to import langflow...")
    try:
        import langflow
        print(f"SUCCESS: langflow version: {langflow.__version__}")
    except Exception as e:
        print(f"ERROR importing langflow: {str(e)}")
        traceback.print_exc()
    
    print("\nTrying to import src.integrations.langflow.react_component...")
    try:
        import src.integrations.langflow.react_component
        print("SUCCESS: Imported src.integrations.langflow.react_component!")
    except Exception as e:
        print(f"ERROR importing react_component: {str(e)}")
        traceback.print_exc()
    
    print("\nTrying to import individual dependencies...")
    
    modules_to_check = [
        "langchain", 
        "langchain_community",
        "langchain_core",
        "langflow.base.agents.agent",
        "langflow.custom",
        "langflow.custom.custom_component.component"
    ]
    
    for module in modules_to_check:
        try:
            print(f"Importing {module}...")
            __import__(module)
            print(f"SUCCESS: {module} imported")
        except Exception as e:
            print(f"ERROR importing {module}: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    debug_imports()
