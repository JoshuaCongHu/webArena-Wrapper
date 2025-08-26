#!/bin/bash
# scripts/entrypoint.sh

set -e

# Function to wait for services
wait_for_service() {
    local host=$1
    local port=$2
    local service=$3
    echo "Waiting for $service at $host:$port..."
    while ! nc -z $host $port; do
        sleep 1
    done
    echo "$service is ready!"
}

# Setup display for headless browser
if [ "$ENABLE_BROWSER" = "true" ]; then
    Xvfb :99 -screen 0 1920x1080x24 &
    export DISPLAY=:99
fi

# Wait for Redis if needed
if [ ! -z "$REDIS_URL" ]; then
    REDIS_HOST=$(echo $REDIS_URL | sed -E 's/redis:\/\/([^:]+).*/\1/')
    wait_for_service ${REDIS_HOST:-redis} 6379 "Redis"
fi

# Parse command
COMMAND=${1:-train}
shift

case $COMMAND in
    train)
        echo "Starting distributed training..."
        exec python3 /workspace/scripts/train_distributed.py "$@"
        ;;
    
    serve)
        echo "Starting API server..."
        exec python3 /workspace/api/server.py "$@"
        ;;
    
    evaluate)
        echo "Running evaluation..."
        exec python3 /workspace/scripts/evaluate.py "$@"
        ;;
    
    jupyter)
        echo "Starting Jupyter notebook..."
        exec jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''
        ;;
    
    tensorboard)
        echo "Starting TensorBoard..."
        exec tensorboard --logdir=/workspace/logs --host=0.0.0.0 --port=6006
        ;;
    
    worker)
        echo "Starting Celery worker..."
        exec celery -A training.tasks worker --loglevel=info "$@"
        ;;
    
    bash)
        exec /bin/bash "$@"
        ;;
    
    test)
        echo "Running tests..."
        exec python3 -m pytest /workspace/tests "$@"
        ;;
    
    *)
        echo "Unknown command: $COMMAND"
        echo "Available commands: train, serve, evaluate, jupyter, tensorboard, worker, bash, test"
        exit 1
        ;;
esac