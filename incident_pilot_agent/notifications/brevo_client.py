"""Brevo transactional-email REST client (https://api.brevo.com/v3/smtp/email)
-- used by agents/notifier.py to send incident notification emails.

Deliberately REST, not SMTP: every other external call in this repo
already goes through httpx (LLM providers, GatewayContextProvider) -- this
follows that same pattern instead of introducing smtplib as a second way
of talking to the outside world.
"""

import httpx

_BREVO_API_URL = "https://api.brevo.com/v3/smtp/email"


class BrevoAPIError(Exception):
    """Raised on a non-2xx response from Brevo's REST API. Callers (see
    agents/notifier.py) must catch this -- a failed notification is an
    ancillary concern and must never crash the investigation pipeline."""


class BrevoEmailClient:
    def __init__(
        self,
        api_key: str,
        sender_email: str,
        *,
        sender_name: str = "Incident Pilot",
        timeout_seconds: float = 10.0,
    ):
        self._api_key = api_key
        self._sender_email = sender_email
        self._sender_name = sender_name
        self._timeout_seconds = timeout_seconds

    async def send_text_email(self, *, to_email: str, subject: str, text_content: str) -> str:
        """Sends a plain-text email via Brevo's transactional email API.
        Returns the response's messageId on success; raises BrevoAPIError
        on any non-2xx response or network failure."""
        payload = {
            "sender": {"email": self._sender_email, "name": self._sender_name},
            "to": [{"email": to_email}],
            "subject": subject,
            "textContent": text_content,
        }
        headers = {"api-key": self._api_key, "Content-Type": "application/json", "Accept": "application/json"}

        try:
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                response = await client.post(_BREVO_API_URL, json=payload, headers=headers)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise BrevoAPIError(f"Brevo API request failed: {exc}") from exc

        return response.json().get("messageId", "")
