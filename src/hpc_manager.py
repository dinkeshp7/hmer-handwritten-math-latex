"""
PARAM Kamrupa HPC Master Control Engine (User: p.dinkesh)

Automates 2FA SSH authentication, ASCII captcha solving, job queue monitoring,
log fetching, Pillow dependency verification, and dataset synchronization.
"""

import sys
import os
import time
import re
import argparse
from pathlib import Path
import paramiko

HOSTNAME = "paramkamrupa.iitg.ac.in"
PORT = 4422
USERNAME = "p.dinkesh"
PASSWORD = "UHrOE39I6VfA"


def parse_ascii_captcha(prompt_text: str) -> str:
    """
    Parses ASCII captcha pattern from PARAM Kamrupa login prompt.
    Example prompt format:
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
        print(f"[Captcha Solver] Solved ASCII Captcha: '{captcha_str}'")
        return captcha_str
    return ""


class ParamKamrupaManager:
    def __init__(self, otp_code: str, username: str = USERNAME, hostname: str = HOSTNAME, port: int = PORT, password: str = PASSWORD):
        self.username = username
        self.hostname = hostname
        self.port = port
        self.password = password
        self.otp_code = otp_code
        self.client = None

    def connect(self) -> bool:
        print(f"Connecting to {self.username}@{self.hostname}:{self.port} using 2FA OTP [{self.otp_code}]...")

        def auth_handler(title, instructions, prompt_list):
            responses = []
            full_text = f"{title} {instructions}"
            captcha_ans = parse_ascii_captcha(full_text)

            for prompt, echo in prompt_list:
                prompt_lower = prompt.lower()
                if any(k in prompt_lower for k in ["type the string", "challenge", "string above", "captcha"]):
                    responses.append(captcha_ans)
                elif any(k in prompt_lower for k in ["verification", "otp", "code"]):
                    responses.append(self.otp_code)
                elif "password" in prompt_lower:
                    responses.append(self.password)
                else:
                    responses.append(self.password)
            return responses

        try:
            trans = paramiko.Transport((self.hostname, self.port))
            trans.start_client()
            trans.auth_interactive(self.username, auth_handler)

            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.client._transport = trans

            print("✅ Successfully authenticated & connected to PARAM Kamrupa!")
            return True
        except Exception as e:
            print(f"❌ Connection Failed: {e}")
            return False

    def run_cmd(self, command: str, print_output: bool = True) -> tuple[str, str, int]:
        if not self.client:
            raise RuntimeError("SSH Client is not connected.")
        stdin, stdout, stderr = self.client.exec_command(command)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        exit_code = stdout.channel.recv_exit_status()
        if print_output:
            print(f"\n--- Output of: {command} ---")
            if out:
                print(out)
            if err:
                print(f"[STDERR]\n{err}")
        return out, err, exit_code

    def close(self):
        if self.client:
            self.client.close()
            print("Disconnected from PARAM Kamrupa.")


def main():
    parser = argparse.ArgumentParser(description="PARAM Kamrupa Master Control Engine")
    parser.add_argument("--otp", type=str, required=True, help="6-digit 2FA OTP code from Authenticator App")
    parser.add_argument("--command", type=str, default="", help="Custom bash command to execute remotely")
    parser.add_argument("--action", type=str, choices=["status", "logs", "fix-pillow", "cmd"], default="status", help="Preset action to perform")
    args = parser.parse_args()

    mgr = ParamKamrupaManager(otp_code=args.otp)
    if not mgr.connect():
        sys.exit(1)

    try:
        if args.action == "status":
            print("\n=== Checking Slurm Queue ===")
            mgr.run_cmd("squeue -u p.dinkesh")
            print("\n=== Directory Contents on /scratch ===")
            mgr.run_cmd("ls -la /scratch/p.dinkesh/hmer_project/logs/")
        elif args.action == "logs":
            print("\n=== Fetching Latest Slurm Job Output Logs ===")
            mgr.run_cmd("ls -t /scratch/p.dinkesh/hmer_project/logs/*.out | head -n 1 | xargs -r cat")
            print("\n=== Fetching Latest Slurm Job Error Logs ===")
            mgr.run_cmd("ls -t /scratch/p.dinkesh/hmer_project/logs/*.err | head -n 1 | xargs -r cat")
        elif args.action == "fix-pillow":
            print("\n=== Installing Pre-compiled Pillow Binary Wheel ===")
            mgr.run_cmd("source /scratch/p.dinkesh/hmer_project/venv/bin/activate && pip install --only-binary=:all: pillow")
        elif args.action == "cmd":
            if not args.command:
                print("Error: --command must be specified for action 'cmd'.")
            else:
                mgr.run_cmd(args.command)
    finally:
        mgr.close()


if __name__ == "__main__":
    main()
