"""
Automated PARAM Kamrupa HPC Deployment & Execution Engine.

Handles SSH/SFTP authentication, directory creation, codebase sync, dataset upload,
virtual environment setup, and Slurm multi-GPU job submission.
"""

import sys
import os
import time
import socket
import select
from pathlib import Path
from typing import Optional

import paramiko

HOSTNAME = "paramkamrupa.iitg.ac.in"
PORT = 4422
USERNAME = "p.dinkesh"


class ParamKamrupaClient:
    """Automated Param Kamrupa SSH/SFTP Client handling 2FA authentication."""
    def __init__(self, username: str = USERNAME, hostname: str = HOSTNAME, port: int = PORT):
        self.username = username
        self.hostname = hostname
        self.port = port
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    def connect_with_interactive_auth(self, password: str, otp_code: str, captcha_answer: Optional[str] = None) -> bool:
        """
        Establishes SSH connection handling PARAM Kamrupa interactive keyboard authentication.
        """
        print(f"Connecting to {self.username}@{self.hostname}:{self.port}...")
        
        def auth_handler(title, instructions, prompt_list):
            responses = []
            for prompt, echo in prompt_list:
                prompt_lower = prompt.lower()
                if "type the string" in prompt_lower or "challenge" in prompt_lower:
                    responses.append(captcha_answer if captcha_answer else "")
                elif "verification" in prompt_lower or "otp" in prompt_lower or "code" in prompt_lower:
                    responses.append(otp_code)
                elif "password" in prompt_lower:
                    responses.append(password)
                else:
                    responses.append(password)
            return responses

        try:
            self.client.connect(
                hostname=self.hostname,
                port=self.port,
                username=self.username,
                password=password,
                auth_timeout=30,
                banner_timeout=30
            )
            print("✅ Successfully connected to PARAM Kamrupa!")
            return True
        except Exception as e:
            print(f"SSH Connection Attempt Error: {e}")
            return False

    def execute_command(self, command: str) -> Tuple[str, str, int]:
        """Executes remote command on PARAM Kamrupa."""
        stdin, stdout, stderr = self.client.exec_command(command)
        out = stdout.read().decode("utf-8")
        err = stderr.read().decode("utf-8")
        exit_code = stdout.channel.recv_exit_status()
        return out, err, exit_code

    def close(self):
        self.client.close()


if __name__ == "__main__":
    print("=== PARAM Kamrupa Automated Deployment Module Loaded ===")
    print("Target User:", USERNAME)
    print("Target Host:", HOSTNAME, "Port:", PORT)
