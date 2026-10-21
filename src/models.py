"""Define and validate data models for function-calling inputs."""
from pydantic import BaseModel, model_validator, ConfigDict
from typing import Self


class Prompt(BaseModel):
    """Represent and validate a user prompt."""

    model_config = ConfigDict(extra="forbid")
    prompt: str

    @model_validator(mode='after')
    def validate_name(self) -> Self:
        """Ensure the prompt key is valid and its value is not empty.

        Returns:
            The validated Prompt instance.

        Raises:
            ValueError: If the key is not 'prompt' or the value is empty.
        """
        if not self.prompt.strip():
            raise ValueError("prompt cannot be empty.")
        return self


class FunctionDefinition(BaseModel):
    """Represent and validate a function's definition."""

    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]

    @model_validator(mode='after')
    def validate_function_definition(self) -> Self:
        """Validate the function name, description,
        parameters, and return type.

        Returns:
            The validated FunctionDefinition instance.

        Raises:
            ValueError: If a required field is empty or a type is invalid.
        """
        valid_types = [
            "string", "number", "integer", "float",
            "boolean", "array", "object"]
        if not self.name.strip():
            raise ValueError("function name cannot be empty")
        if not self.description.strip():
            raise ValueError("function description cannot be empty")
        for key in self.parameters:
            if self.parameters[key]["type"] not in valid_types:
                raise ValueError(
                    "'number' and 'string' are the only"
                    "valid types."
                    )
            elif not key.strip():
                raise ValueError(
                    "keys and values cannot be empty "
                    "in functions definitions."
                    )
            elif "type" not in self.parameters[key]:
                raise ValueError("key 'type' is mandatory for each parameter.")
            elif not self.parameters[key]["type"].strip():
                raise ValueError(
                    "keys and values cannot be empty"
                    " in functions definitions.")

        if self.returns["type"] not in valid_types:
            raise ValueError(
                "'number' 'string' and 'boolean' are the only"
                "valid types."
                )
        return self
