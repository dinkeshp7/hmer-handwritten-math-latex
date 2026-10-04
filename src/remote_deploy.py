"""
PARAM Kamrupa Remote Automated Deployer for p.dinkesh with ASCII Captcha Solver & 2FA Handler.
"""

import sys
import os
import time
import re
import socket
from pathlib import Path
import paramiko

HOSTNAME = "paramkamrupa.iitg.ac.in"
PORT = 4422
USERNAME = "p.dinkesh"
PASSWORD = "UHrOE39I6VfA"
OTP_CODE = "276513"


def parse_ascii_captcha(prompt_text: str) -> str:
    """
    Parses ASCII captcha pattern like:
      _   _   _   _   _   _   _   _
     / \ / \ / \ / \ / \ / \ / \ / \
    ( g | u | P | b | H | J | A | O )
     \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/
    Extracts 'guPbHJAO'.
    """
    match = re.search(r"\(\s*([a-zA-Z0-9\s\|]+)\s*\)", prompt_text)
    if match:
        raw_chars = match.group(1)
        chars = [c.strip() for c in raw_chars.split("|") if c.strip()]
        captcha_str = "".join(chars)
        print(f"Parsed Captcha Challenge: '{captcha_str}'")
        return captcha_str
    return ""


def run_deployment():
    print(f"Connecting to {USERNAME}@{HOSTNAME}:{PORT} with OTP {OTP_CODE}...")

    def auth_handler(title, instructions, prompt_list):
        responses = []
        full_text = title + " " + instructions
        captcha_ans = parse_ascii_captcha(full_text)
        
        for prompt, echo in prompt_list:
            prompt_lower = prompt.lower()
            if "type the string" in prompt_lower or "challenge" in prompt_lower or "string above" in prompt_lower:
                responses.append(captcha_ans)
            elif "verification" in prompt_lower or "otp" in prompt_lower or "code" in prompt_lower:
                responses.append(OTP_CODE)
            elif "password" in prompt_lower:
                responses.append(PASSWORD)
            else:
                responses.append(PASSWORD)
        return responses

    try:
        trans = paramiko.Transport((HOSTNAME, PORT))
        trans.start_client()
        trans.auth_interactive(USERNAME, auth_handler)
        
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client._transport = trans
        
        print("✅ Connected to PARAM Kamrupa!")
        stdin, stdout, stderr = client.exec_command("echo '=== Logged in as p.dinkesh ==='; pwd; whoami")
        print(stdout.read().decode("utf-8"))
        client.close()
    except Exception as e:
        print(f"Auth Error: {e}")


if __name__ == "__main__":
    run_deployment()
