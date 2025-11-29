import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app:app",          # 👈 app.py is in the SAME folder as this file
        host="0.0.0.0",
        port=port,
        access_log=False,
        timeout_keep_alive=65,
    )
