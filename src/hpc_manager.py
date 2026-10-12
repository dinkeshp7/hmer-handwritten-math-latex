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
    r"""
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
            prompts_joined = " ".join([p[0] if isinstance(p, tuple) else str(p) for p in prompt_list])
            full_text = f"{title} {instructions} {prompts_joined}"
            captcha_ans = parse_ascii_captcha(full_text)

            for prompt, echo in prompt_list:
                prompt_text = prompt[0] if isinstance(prompt, tuple) else str(prompt)
                prompt_lower = prompt_text.lower()
                print(f"[SSH Auth Prompt] {prompt_text.strip()}")
                if any(k in prompt_lower for k in ["type the string", "challenge", "string above", "captcha"]):
                    print(f"   --> Sending Solved Captcha: '{captcha_ans}'")
                    responses.append(captcha_ans)
                elif any(k in prompt_lower for k in ["verification", "otp", "code"]):
                    print(f"   --> Sending OTP Code: '{self.otp_code}'")
                    responses.append(self.otp_code)
                elif "password" in prompt_lower:
                    print("   --> Sending Password")
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

    def upload_file(self, local_path: str, remote_path: str) -> bool:
        if not self.client:
            raise RuntimeError("SSH Client is not connected.")
        try:
            print(f"Uploading {local_path} -> {remote_path} via SFTP...")
            sftp = self.client.open_sftp()
            sftp.put(local_path, remote_path)
            sftp.close()
            print("✅ File uploaded successfully via SFTP!")
            return True
        except Exception as e:
            print(f"❌ SFTP Upload Failed: {e}")
            return False

    def upload_directory(self, local_dir: str, remote_dir: str) -> bool:
        if not self.client:
            raise RuntimeError("SSH Client is not connected.")
        try:
            sftp = self.client.open_sftp()
            local_path = Path(local_dir).resolve()
            print(f"Uploading directory recursively: {local_path} -> {remote_dir}...")
            
            # Ensure base remote directory exists
            try:
                sftp.mkdir(remote_dir)
            except IOError:
                pass
                
            for root, dirs, files in os.walk(local_path):
                rel_path = Path(root).relative_to(local_path)
                if str(rel_path) != ".":
                    target_remote_dir = f"{remote_dir}/{rel_path}".replace("\\", "/")
                    try:
                        sftp.mkdir(target_remote_dir)
                    except IOError:
                        pass
                else:
                    target_remote_dir = remote_dir
                
                for file_name in files:
                    local_file_path = os.path.join(root, file_name)
                    remote_file_path = f"{target_remote_dir}/{file_name}".replace("\\", "/")
                    print(f"  --> Uploading {file_name} ({os.path.getsize(local_file_path) / (1024*1024):.1f} MB)...")
                    sftp.put(local_file_path, remote_file_path)
            
            sftp.close()
            print("✅ Directory uploaded successfully via SFTP!")
            return True
        except Exception as e:
            print(f"❌ SFTP Directory Upload Failed: {e}")
            return False

    def close(self):
        if self.client:
            self.client.close()
            print("Disconnected from PARAM Kamrupa.")


def main():
    parser = argparse.ArgumentParser(description="PARAM Kamrupa Master Control Engine")
    parser.add_argument("--otp", type=str, required=True, help="6-digit 2FA OTP code from Authenticator App")
    parser.add_argument("--command", type=str, default="", help="Custom bash command to execute remotely")
    parser.add_argument("--action", type=str, choices=["status", "logs", "fix-pillow", "upload", "cmd"], default="status", help="Preset action to perform")
    parser.add_argument("--file", type=str, default="hmer_codebase.tar.gz", help="Local file or directory to upload")
    args = parser.parse_args()

    mgr = ParamKamrupaManager(otp_code=args.otp)
    if not mgr.connect():
        sys.exit(1)

    try:
        if args.action == "upload":
            local_target = Path(args.file).resolve()
            if not local_target.exists():
                print(f"Error: Local path '{local_target}' not found.")
                return
            if local_target.is_dir():
                remote_dir = f"/scratch/p.dinkesh/hmer_project/models/{local_target.name}"
                mgr.upload_directory(str(local_target), remote_dir)
            else:
                remote_file = f"/scratch/p.dinkesh/hmer_project/{local_target.name}"
                if mgr.upload_file(str(local_target), remote_file):
                    if local_target.name.endswith(".tar.gz"):
                        print("\n=== Extracting Codebase Tarball on PARAM Kamrupa ===")
                        mgr.run_cmd("cd /scratch/p.dinkesh/hmer_project && tar -xzf hmer_codebase.tar.gz")
                        print("\n=== Directory Contents After Unpacking ===")
                        mgr.run_cmd("ls -la /scratch/p.dinkesh/hmer_project/")
        elif args.action == "status":
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
