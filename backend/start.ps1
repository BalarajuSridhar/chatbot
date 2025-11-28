$env:PORT = if ($env:PORT) { $env:PORT } else { "8000" }
uvicorn app:app --host 0.0.0.0 --port $env:PORT