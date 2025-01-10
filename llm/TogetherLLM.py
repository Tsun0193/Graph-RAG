import os
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    LLMMetadata
)
from llama_index.core.llms.callbacks import llm_completion_callback
from together import Together
from dotenv import load_dotenv
from typing import Optional, List, Any

os.chdir("../")
load_dotenv()

client = Together()

def get_response(self, query):
    response = client.chat.completions.create(
        model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K",
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": query
            }
        ],
        max_tokens = 512,
        temperature = 0,
        top_p = 0.7,
        top_k = 50,
        repetition_penalty = 1,
        stop = ["<|eot_id|>", "<|eom_id|>"],
        stream = False
    )

    return response.choices[0].message["content"]

class CustomLLM(CustomLLM):
    """
    Custom LLM class using Together AI API.
    """
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K"

    def __init__(self, model: str = None):
        """
        Initializes the Custom LLM with the specified model.
        """
        super().__init__(model or self.model)
        self.model = model or self.model
        print(
            f"""
            Custom LLM initialized with model: {self.model}
            """
        )

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
                 history: Optional[List[dict]] = None,
                 **kwargs) -> CompletionResponse:
        """
        Completion endpoint for the LLM.
        """
        # TODO: Implement the completion logic here
        pass


if __name__ == "__main__":
    pass