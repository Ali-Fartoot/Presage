#!/bin/bash
curl -X POST "http://localhost:8000/presage/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@./example/test.jpg"