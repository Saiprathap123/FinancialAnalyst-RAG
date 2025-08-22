from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_experimental.tools.python.tool import PythonREPLTool

def get_web_search_tool():
    """
    Returns a DuckDuckGo search tool for web queries.
    """
    return DuckDuckGoSearchRun(name="Web_Search", description="Search the web for latest information using DuckDuckGo.")

def get_python_repl_tool():
    """
    Returns a Python REPL tool for executing Python code snippets.
    """
    return PythonREPLTool()
