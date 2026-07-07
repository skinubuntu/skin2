#!/bin/bash

cd "$(dirname "$0")"

source venv/bin/activate

python3 app.py &

until curl -s http://127.0.0.1:5000 > /dev/null; do
    sleep 1
done

xdg-open http://127.0.0.1:5000