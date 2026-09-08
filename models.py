from pydantic import BaseModel, model_validator
from typing import Self


class Prompt(BaseModel):
    prompt_key: str
    prompt_value: str

    @model_validator(mode='after')
    def validate_name(self) -> Self:
        if self.prompt_key != "prompt":
            raise ValueError("function_calling's key should be 'prompt'")
        return self


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]

    @model_validator(mode='after')
    def validate_function_definition(self) -> Self:
        valid_types = ["string", "number"]
        for key in self.parameters:
            if self.parameters[key]["type"] not in valid_types:
                raise ValueError(
                    "'number' and 'string' are the only"
                    "valid types."
                    )
        if self.returns["type"] not in valid_types:
            raise ValueError(
                "'number' and 'string' are the only"
                "valid types."
                )
        return self
