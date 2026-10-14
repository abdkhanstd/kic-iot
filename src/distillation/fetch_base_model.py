# Author: abdkhan
"""Fetch a tiny-LLM base checkpoint via ModelScope, since huggingface.co is
unreachable from this network and hf-mirror.com's /resolve/ endpoint
redirects back to the blocked origin instead of proxying downloads.
"""
import argparse

from modelscope import snapshot_download

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id", help="e.g. Qwen/Qwen2.5-0.5B-Instruct")
    args = ap.parse_args()
    path = snapshot_download(args.model_id)
    print(path)
