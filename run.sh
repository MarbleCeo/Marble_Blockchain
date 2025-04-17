#!/bin/bash

export DEBUG=1
uvicorn main:app --host 127.0.0.1 --port 8000 --workers 2 --timeout-keep-alive 30 --log-config logging.conf

