#!/usr/bin/env python3
# scripts/deploy_runpod.py
import os
import sys
import json
import time
import argparse
import logging
from typing import Dict, Any, Optional
from pathlib import Path

import requests
import runpod

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RunPodDeployment:
    """Manage RunPod deployments for training"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        runpod.api_key = api_key
        
    def create_training_pod(self, config: Dict[str, Any]) -> str:
        """Create a new training pod on RunPod"""
        
        # Build pod configuration
        pod_config = {
            "name": f"webarena-mas-{config['method']}-{int(time.time())}",
            "image": config.get("image", "webarena-mas:latest"),
            "gpu_type_id": self._get_gpu_type_id(config.get("gpu_type", "NVIDIA RTX A6000")),
            "gpu_count": config.get("gpu_count", 1),
            "volume_in_gb": config.get("volume_gb", 100),
            "container_disk_in_gb": 50,
            "min_memory_in_gb": config.get("memory_gb", 32),
            "docker_args": self._build_docker_command(config),
            "env": self._build_environment(config),
            "ports": "8000/http,6006/http,8888/http,22/tcp",
            "volume_mount_path": "/workspace/persistent",
            "start_jupyter": config.get("start_jupyter", False),
            "start_ssh": config.get("start_ssh", True)
        }
        
        logger.info(f"Creating pod with configuration: {json.dumps(pod_config, indent=2)}")
        
        try:
            response = runpod.create_pod(**pod_config)
            pod_id = response["id"]
            logger.info(f"Successfully created pod: {pod_id}")
            
            # Wait for pod to be ready
            self._wait_for_pod_ready(pod_id)
            
            return pod_id
            
        except Exception as e:
            logger.error(f"Failed to create pod: {e}")
            raise
    
    def _get_gpu_type_id(self, gpu_type: str) -> str:
        """Map GPU type name to RunPod GPU type ID"""
        gpu_mapping = {
            "NVIDIA RTX A6000": "NVIDIA RTX A6000",
            "NVIDIA RTX A5000": "NVIDIA RTX A5000",
            "NVIDIA RTX 3090": "NVIDIA GeForce RTX 3090",
            "NVIDIA A100": "NVIDIA A100 80GB PCIe",
            "NVIDIA A100-40GB": "NVIDIA A100-PCIE-40GB",
            "NVIDIA A40": "NVIDIA A40",
            "NVIDIA RTX 4090": "NVIDIA GeForce RTX 4090"
        }
        return gpu_mapping.get(gpu_type, gpu_type)
    
    def _build_docker_command(self, config: Dict[str, Any]) -> str:
        """Build Docker command for training"""
        cmd_parts = [
            "train",
            "--method", config.get("method", "p3o"),
            "--episodes", str(config.get("episodes", 10000)),
            "--batch-size", str(config.get("batch_size", 32)),
            "--num-gpus", str(config.get("gpu_count", 1)),
            "--checkpoint-dir", "/workspace/persistent/checkpoints",
            "--data-path", "/workspace/persistent/data",
            "--log-dir", "/workspace/persistent/logs"
        ]
        
        if config.get("distributed", True) and config.get("gpu_count", 1) > 1:
            cmd_parts.append("--distributed")
        
        if config.get("use_llm_orchestrator", True):
            cmd_parts.extend([
                "--use-llm-orchestrator",
                "--llm-model", config.get("llm_model", "gpt-4-turbo")
            ])
        
        if config.get("enable_replanning", True):
            cmd_parts.append("--enable-replanning")
        
        if config.get("use_wandb", True):
            cmd_parts.extend([
                "--use-wandb",
                "--run-name", config.get("run_name", "runpod-experiment")
            ])
        
        if config.get("resume"):
            cmd_parts.extend(["--resume", config["resume"]])
        
        return " ".join(cmd_parts)
    
    def _build_environment(self, config: Dict[str, Any]) -> list:
        """Build environment variables for pod"""
        env_vars = []
        
        # API Keys
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "WANDB_API_KEY"]:
            value = config.get(key) or os.environ.get(key)
            if value:
                env_vars.append({"key": key, "value": value})
        
        # Training configuration
        env_vars.extend([
            {"key": "TRAINING_MODE", "value": "distributed" if config.get("distributed", True) else "single"},
            {"key": "CUDA_VISIBLE_DEVICES", "value": ",".join(map(str, range(config.get("gpu_count", 1))))},
            {"key": "WANDB_PROJECT", "value": config.get("wandb_project", "webarena-mas")},
            {"key": "PYTHONPATH", "value": "/workspace:/workspace/mas_webarena"}
        ])
        
        return env_vars
    
    def _wait_for_pod_ready(self, pod_id: str, timeout: int = 300):
        """Wait for pod to be ready"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            pod = runpod.get_pod(pod_id)
            
            if pod["status"] == "RUNNING":
                logger.info(f"Pod {pod_id} is ready")
                return
            elif pod["status"] == "FAILED":
                raise RuntimeError(f"Pod {pod_id} failed to start")
            
            logger.info(f"Waiting for pod to be ready... Status: {pod['status']}")
            time.sleep(10)
        
        raise TimeoutError(f"Pod {pod_id} did not become ready within {timeout} seconds")
    
    def get_pod_status(self, pod_id: str) -> Dict[str, Any]:
        """Get current status of a pod"""
        pod = runpod.get_pod(pod_id)
        return {
            "id": pod_id,
            "status": pod["status"],
            "gpu_type": pod.get("machine", {}).get("gpu_type"),
            "gpu_count": pod.get("machine", {}).get("gpu_count"),
            "runtime": pod.get("runtime"),
            "cost_per_hour": pod.get("cost_per_hour")
        }
    
    def stream_logs(self, pod_id: str):
        """Stream logs from a running pod"""
        logger.info(f"Streaming logs from pod {pod_id}")
        
        for log in runpod.stream_logs(pod_id):
            print(log.strip())
    
    def stop_pod(self, pod_id: str):
        """Stop a running pod"""
        logger.info(f"Stopping pod {pod_id}")
        runpod.stop_pod(pod_id)
    
    def terminate_pod(self, pod_id: str):
        """Terminate a pod"""
        logger.info(f"Terminating pod {pod_id}")
        runpod.terminate_pod(pod_id)
    
    def list_pods(self) -> list:
        """List all pods"""
        pods = runpod.get_pods()
        return [{
            "id": pod["id"],
            "name": pod["name"],
            "status": pod["status"],
            "gpu_type": pod.get("machine", {}).get("gpu_type"),
            "created": pod.get("created_at")
        } for pod in pods]

def load_training_config(config_file: str) -> Dict[str, Any]:
    """Load training configuration from file"""
    config_path = Path(config_file)
    
    if config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            return json.load(f)
    elif config_path.suffix in ['.yaml', '.yml']:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")

def main():
    parser = argparse.ArgumentParser(description='Deploy training to RunPod')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Create pod command
    create_parser = subparsers.add_parser('create', help='Create a new training pod')
    create_parser.add_argument('--config', type=str, required=True,
                             help='Configuration file (JSON or YAML)')
    create_parser.add_argument('--api-key', type=str, 
                             default=os.environ.get('RUNPOD_API_KEY'),
                             help='RunPod API key')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Get pod status')
    status_parser.add_argument('pod_id', type=str, help='Pod ID')
    status_parser.add_argument('--api-key', type=str,
                             default=os.environ.get('RUNPOD_API_KEY'))
    
    # Logs command
    logs_parser = subparsers.add_parser('logs', help='Stream pod logs')
    logs_parser.add_argument('pod_id', type=str, help='Pod ID')
    logs_parser.add_argument('--api-key', type=str,
                           default=os.environ.get('RUNPOD_API_KEY'))
    
    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop a pod')
    stop_parser.add_argument('pod_id', type=str, help='Pod ID')
    stop_parser.add_argument('--api-key', type=str,
                           default=os.environ.get('RUNPOD_API_KEY'))
    
    # Terminate command
    terminate_parser = subparsers.add_parser('terminate', help='Terminate a pod')
    terminate_parser.add_argument('pod_id', type=str, help='Pod ID')
    terminate_parser.add_argument('--api-key', type=str,
                                default=os.environ.get('RUNPOD_API_KEY'))
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all pods')
    list_parser.add_argument('--api-key', type=str,
                           default=os.environ.get('RUNPOD_API_KEY'))
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if not args.api_key:
        logger.error("RunPod API key is required. Set RUNPOD_API_KEY environment variable or use --api-key")
        return
    
    deployment = RunPodDeployment(args.api_key)
    
    if args.command == 'create':
        config = load_training_config(args.config)
        pod_id = deployment.create_training_pod(config)
        print(f"Created pod: {pod_id}")
        
        # Optionally stream logs
        if config.get("stream_logs", False):
            deployment.stream_logs(pod_id)
    
    elif args.command == 'status':
        status = deployment.get_pod_status(args.pod_id)
        print(json.dumps(status, indent=2))
    
    elif args.command == 'logs':
        deployment.stream_logs(args.pod_id)
    
    elif args.command == 'stop':
        deployment.stop_pod(args.pod_id)
        print(f"Stopped pod: {args.pod_id}")
    
    elif args.command == 'terminate':
        deployment.terminate_pod(args.pod_id)
        print(f"Terminated pod: {args.pod_id}")
    
    elif args.command == 'list':
        pods = deployment.list_pods()
        for pod in pods:
            print(f"{pod['id']}: {pod['name']} ({pod['status']}) - {pod['gpu_type']}")

if __name__ == "__main__":
    main()