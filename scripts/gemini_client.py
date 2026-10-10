"""Standalone Vertex AI Gemini text/vision client for Scale-Plan.

Mirrors LABOR-Agent's client so both planners talk to RoboPARA-2.0 the same
way: ``call(prompt, image_path)`` sends text plus an optional scene image.
If Vertex reports the primary model as unavailable (404), the client switches
to the fallback model for the rest of its lifetime.
"""

import argparse
import mimetypes
import os
from pathlib import Path

from google import genai
from google.genai import errors, types
from google.oauth2 import service_account

# Match RoboPARA_new/src/RAG.py; credentials remain outside the repository.
DEFAULT_CREDENTIALS = Path.home() / ".config/robopara/p-150gk23k-718446bc2ebd.json"
DEFAULT_PROJECT = "p-150gk23k"
DEFAULT_MODEL = "gemini-3.7-flash"
DEFAULT_FALLBACK_MODEL = "gemini-3.8-flash"
VERTEX_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)


class GeminiClient:
    """Explicit options override environment variables and reference defaults."""

    def __init__(self, api_path=None, model=None, project=None, location=None,
                 credentials=None, fallback_model=None):
        self.model = model or os.getenv("VERTEX_MODEL") or DEFAULT_MODEL
        self.fallback_model = (fallback_model or os.getenv("VERTEX_FALLBACK_MODEL")
                               or DEFAULT_FALLBACK_MODEL)
        self.project = project or os.getenv("GOOGLE_CLOUD_PROJECT") or DEFAULT_PROJECT
        self.location = location or os.getenv("GOOGLE_CLOUD_LOCATION") or "global"
        self.credentials_path = Path(
            api_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or DEFAULT_CREDENTIALS
        ).expanduser()
        if credentials is None:
            if not self.credentials_path.is_file():
                raise FileNotFoundError(
                    f"Vertex service-account file not found: {self.credentials_path}. "
                    "Set GOOGLE_APPLICATION_CREDENTIALS or pass api_path."
                )
            credentials = service_account.Credentials.from_service_account_file(
                str(self.credentials_path), scopes=VERTEX_SCOPES,
            )
        self.client = genai.Client(
            vertexai=True, project=self.project, location=self.location,
            credentials=credentials,
            http_options=types.HttpOptions(timeout=120_000),
        )

    def call(self, prompt: str, image_path=None, temperature=0.0) -> str:
        """Send text with an optional local scene image; return model text."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        contents = prompt
        if image_path is not None:
            path = Path(image_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Scene image not found: {path}")
            mime_type = mimetypes.guess_type(path.name)[0]
            if mime_type not in {"image/png", "image/jpeg", "image/webp"}:
                raise ValueError("Scene image must be PNG, JPEG or WebP")
            contents = [prompt, types.Part.from_bytes(
                data=path.read_bytes(), mime_type=mime_type,
            )]
        config = types.GenerateContentConfig(temperature=temperature)
        try:
            response = self.client.models.generate_content(
                model=self.model, contents=contents, config=config,
            )
        except errors.ClientError as error:
            if error.code != 404 or self.model == self.fallback_model:
                raise
            print(f"[gemini] {self.model} unavailable; switching to {self.fallback_model}")
            self.model = self.fallback_model
            response = self.client.models.generate_content(
                model=self.model, contents=contents, config=config,
            )
        if not response.text or not response.text.strip():
            raise RuntimeError("Gemini returned no text; request may have been blocked")
        return response.text

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="Reply with exactly: GEMINI_OK")
    parser.add_argument("--image", help="Local PNG, JPEG or WebP scene image")
    parser.add_argument("--model")
    parser.add_argument("--fallback-model")
    parser.add_argument("--project")
    parser.add_argument("--location")
    parser.add_argument("--credentials", help="Vertex service-account JSON path")
    args = parser.parse_args()
    with GeminiClient(api_path=args.credentials, model=args.model,
                      project=args.project, location=args.location,
                      fallback_model=args.fallback_model) as client:
        print(client.call(args.prompt, image_path=args.image))


if __name__ == "__main__":
    main()
