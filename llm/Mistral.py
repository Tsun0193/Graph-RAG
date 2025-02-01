import os
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse, CompletionResponseGen,
    LLMMetadata
)
from llama_index.core.llms.callbacks import llm_completion_callback
from typing import Optional, List, Any, Dict
from pydantic import Field
from ctransformers import AutoModelForCausalLM


class Mistral(CustomLLM):
    """
    Custom LLM class using Mistral Instruct model.
    """

    model: str = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
    llm: Optional[AutoModelForCausalLM] = Field(default=None, init=False, repr=False)

    def __init__(self,
                 **kwargs) -> None:
        """
        Initialize the Mistral class.
        """
        super().__init__(**kwargs)
        object.__setattr__(self, "llm", AutoModelForCausalLM.from_pretrained(
            self.model, 
            model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf", 
            model_type="mistral"
        ))
        print(f"Mistral LLM initialized.")

    def get_response(self, query,
                     history: Optional[List[Dict[str, Any]]] = None,
                     **kwargs) -> str:
        messages = ""
        if history:
            for item in history:
                if item["role"] == "user":
                    messages += f"<s>[INST] {item['content']} [/INST]</s>\n"
                elif item["role"] == "assistant":
                    messages += f"{item['content']}\n"
                else:
                    continue # Ignore system messages

        messages += f"<s>[INST] {query} [/INST]</s>"

        try:
            response = self.llm(messages)
        except Exception as e:
            raise Exception(f"Error: {str(e)}")
        
        return response

    @property
    def metadata(self) -> LLMMetadata:
        """
        Return the metadata of the Mistral LLM.
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
        Complete the prompt using the Mistral LLM.
        """
        response = self.get_response(prompt, history=history, model=self.llm, **kwargs)
        additional_kwargs = {
            "model": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
        }
        return CompletionResponse(
            text=response,
            additional_kwargs=additional_kwargs
        )
    
    @llm_completion_callback()
    def stream_complete(self, prompt, formatted = False, **kwargs):
        """
        Stream completion endpoint for the Mistral LLM.
        """
        response = self.get_response(prompt, model = self.llm, stream = True, **kwargs)
        accumulated_text = ""
        for char in response:
            accumulated_text += char
            if char == "\n":
                yield CompletionResponse(text = accumulated_text)


if __name__ == "__main__":
    prompt = "What is the capital of France?"
    mistral = Mistral()
    response = mistral.complete(prompt)
    print(response.text)