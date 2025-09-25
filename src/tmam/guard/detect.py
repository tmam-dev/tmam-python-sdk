"""
guard detect
"""

import requests
from sdk.python.src.tmam.guard.utils import JsonOutput
from tmam.__helpers import get_env_variable


class Detect:
    """
    A comprehensive class to detect prompt injections, valid/invalid topics, and sensitive topics using LLM or custom rules.

    Attributes:
        input (Optional[str]): The name of the LLM provider.
        output (Optional[str]): The API key for authenticating with the LLM.
    """

    def __init__(self):
        """
        Initializes the All class with specified settings, last guard prompt id for output.
        """
        self.last_guard_prompt_id = None
        self.guardrail_id = None
        self.name = None
        self.user_id = None

    def input(
        self,
        prompt: str,
        guardrail_id: str,
        name: str | None = None,
        user_id: str | None = None,
    ) -> JsonOutput:
        """
        Retrieve and returns the result from Tmam Guardrail.

        Args:
            prompt (str): The text of your prompt.
            guardrail_id (str): The guardrail ID for authenticating with the server.
            name (Optional[str]): The name of the guardrail for indentify purposes.
            user_id (Optional[str]): The user id of your prompt user.
        """

        self.guardrail_id = guardrail_id
        self.name = name
        self.user_id = user_id

        # Validate and set the base URL
        env_url = get_env_variable(
            "TMAM_URL",
            "url",
            "Missing Tmam URL: Provide as arg or set TMAM_URL env var.",
        )

        # Validate and set the API key
        env_pk_key = get_env_variable(
            "TMAM_PUBLIC_KEY",
            "public_key",
            "Missing Public key: Provide as arg or set TMAM_PUBLIC_KEY env var.",
        )
        env_sk_key = get_env_variable(
            "TMAM_SECRET_KEY",
            "secrect_key",
            "Missing Secret key: Provide as arg or set TMAM_SECRET_KEY env var.",
        )

        endpoint = env_url + "/guardrail/detect"

        payload = {
            "guardrailId": guardrail_id,
            "promptUserId": "" if user_id is None else user_id,
            "prompt": prompt,
            "isInput": True,
            # "guardPromptId": Null
        }

        # Prepare headers
        headers = {
            "X-Public-Key": env_pk_key,
            "X-Secret-Key": env_sk_key,
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                endpoint, json=payload, headers=headers, timeout=120
            )
            response.raise_for_status()
            json = response.json()

            self.last_guard_prompt_id = json["guardPromptId"]

            return json["result"]
        except requests.RequestException as error:
            return None

    def output(self, prompt: str) -> JsonOutput:
        """
        Retrieve and returns the result from Tmam Guardrail.

        Args:
            prompt (str): The text of your prompt.
            guardrail_id (Optional[str]): The guardrail ID for authenticating with the server.
            name (Optional[str]): The name of the guardrail for indentify purposes.
            user_id (Optional[str]): The user id of your prompt user.
        """

        if self.last_guard_prompt_id is None or self.guardrail_id is None:
            raise ValueError("make sure input is defined")

        # Validate and set the base URL
        env_url = get_env_variable(
            "TMAM_URL",
            "url",
            "Missing Tmam URL: Provide as arg or set TMAM_URL env var.",
        )

        # Validate and set the API key
        env_pk_key = get_env_variable(
            "TMAM_PUBLIC_KEY",
            "public_key",
            "Missing Public key: Provide as arg or set TMAM_PUBLIC_KEY env var.",
        )
        env_sk_key = get_env_variable(
            "TMAM_SECRET_KEY",
            "secrect_key",
            "Missing Secret key: Provide as arg or set TMAM_SECRET_KEY env var.",
        )

        endpoint = env_url + "/guardrail/detect"

        usrid = ""

        if self.user_id is not None and len(self.user_id) != 0:
            usrid = self.user_id

        payload = {
            "guardrailId": self.guardrail_id,
            "promptUserId": usrid,
            "prompt": prompt,
            "isInput": False,
            "guardPromptId": self.last_guard_prompt_id,
        }

        # Prepare headers
        headers = {
            "X-Public-Key": env_pk_key,
            "X-Secret-Key": env_sk_key,
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                endpoint, json=payload, headers=headers, timeout=120
            )
            response.raise_for_status()
            json = response.json()

            return json["result"]
        except requests.RequestException as error:
            return None
