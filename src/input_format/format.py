from abc import ABC, abstractmethod


class Format(ABC):

    @abstractmethod
    def get_prompt(self, example: dict, is_train: bool) -> str:
        raise NotImplementedError


class RawFormat(Format):

    format_name = "raw"

    def __init__(self, text_field: str):

        self.text_field = text_field

    def get_prompt(self, example: dict, is_train: bool) -> str:

        text = example[self.text_field]

        return text


class InstructionFormat(ABC):

    def __init__(self, instruction_field: str, response_field: str):

        self.instruction_field = instruction_field
        self.response_field = response_field

    @abstractmethod
    def get_prompt(self, example: dict, is_train: bool) -> str:
        raise NotImplementedError


class AlpacaFormat(InstructionFormat):

    format_name = "alpaca"

    def __init__(self, instruction_field: str, response_field: str, context_field: str = None):

        super().__init__(instruction_field, response_field)
        self.context_field = context_field

    def get_prompt(self, example: dict, is_train: bool) -> str:

        instruction = example[self.instruction_field]
        response = example[self.response_field] if is_train else ""
        context = example.get(self.context_field, "") if self.context_field else ""

        if context != "":

            return "Di seguito è riportata un'istruzione che descrive un'attività, abbinata ad un input che fornisce ulteriore informazione. " \
                "Scrivi una risposta che soddisfi adeguatamente la richiesta.\n\n" \
                f"### Istruzione:\n{instruction}\n\n### Input:\n{context}\n\n### Risposta:\n{response}"

        else:

            return "Di seguito è riportata un'istruzione che descrive un'attività. " \
                f"Scrivi una risposta che soddisfi adeguatamente la richiesta.\n\n" \
                f"### Istruzione:\n{instruction}\n\n### Risposta:\n{response}"
