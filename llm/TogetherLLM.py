import os
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse, CompletionResponseGen,
    LLMMetadata
)
from llama_index.core.llms.callbacks import llm_completion_callback
from together import Together
from dotenv import load_dotenv
from typing import Optional, List, Any, Dict

load_dotenv()

client = Together()

def get_response(query, 
                 history: Optional[List[Dict[str, Any]]] = None,
                 model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K",
                 **kwargs) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": query
        }
    ]
    if history:
        for item in history:
            if item["role"] == "user":
                messages.append({
                    "role": "user",
                    "content": item["content"]
                })
            elif item["role"] == "assistant":
                messages.append({
                    "role": "assistant",
                    "content": item["content"]
                })
            else:
                continue # Ignore system messages

    try:
        response = client.chat.completions.create(
            model = model,
            messages = messages,
            max_tokens = kwargs.get("max_tokens", 512),
            temperature = kwargs.get("temperature", 0.0),
            top_p = kwargs.get("top_p", 0.7),
            top_k = kwargs.get("top_k", 50),
            repetition_penalty = kwargs.get("repetition_penalty", 1.0),
            stop = ["<|eot_id|>", "<|eom_id|>"],
            stream = kwargs.get("stream", False)
        )
    except Exception as e:
        raise Exception(f"Error: {str(e)}")
    
    return response.choices[0].message.content

class TogetherLLM(CustomLLM):
    """
    Custom LLM class using Together AI API.
    """
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K"

    def __init__(self, model: str = None,
                 **kwargs) -> None:
        """
        Initializes the Custom LLM with the specified model.
        """
        super().__init__(**kwargs)
        self.model = model or self.model
        print(f"Custom LLM initialized with model: {self.model}")

    @property
    def metadata(self) -> LLMMetadata:
        """
        Returns the metadata for the LLM.
        """
        return LLMMetadata(
            num_output=1,
            model_name=self.model
        )
    
    @llm_completion_callback()
    def complete(self, prompt: str,
                 history: Optional[List[Dict[str, Any]]] = None,
                 **kwargs) -> CompletionResponse:
        """
        Completion endpoint for the LLM.
        """
        # TODO: Implement the completion logic here

        try:
            response = get_response(prompt, history, self.model, **kwargs)
        except Exception as e:
            response = str(f"Error: {e}")

        additional_kwargs = {
            "model": self.model
        }
        for key, value in kwargs.items():
            additional_kwargs[key] = value

        return CompletionResponse(
            text=response,
            additional_kwargs=additional_kwargs
        )

    @llm_completion_callback()
    def stream_complete(self, prompt, formatted = False, **kwargs):
        """
        Stream completion endpoint for the LLM.
        """
        try:
            response = get_response(prompt, model = self.model, stream = True, **kwargs)
        except Exception as e:
            yield CompletionResponse(text = "", delta = f"Error: {str(e)}")

        accumulated_text = ""
        for char in response:
            accumulated_text += char
            if char == "\n":
                yield CompletionResponse(text = accumulated_text)
                accumulated_text = ""

        if accumulated_text:
            yield CompletionResponse(text = accumulated_text) # Yield the remaining text

if __name__ == "__main__":
    pass