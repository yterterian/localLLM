# Local LLM System & Open-WebUI Setup Checklist

Follow these steps to resolve the issues and ensure all services run correctly:

## 1. Main System Dependencies
- Open a terminal in your project root.
- Activate your main environment:
  ```
  .venv\Scripts\activate
  ```
- Install all dependencies:
  ```
  pip install -r requirements.txt
  ```

## 2. Context Server Dependencies
- The context server uses the same environment and requirements as the main router.
- If you see `ModuleNotFoundError`, ensure you are running both servers from an activated `.venv` and dependencies are installed.

## 3. Open-WebUI Installation
- Activate your webui environment:
  ```
  .webui-venv\Scripts\activate
  ```
- Install Open-WebUI (replace with the correct command from their documentation, e.g.):
  ```
  pip install open-webui
  ```
- Confirm that `.webui-venv\Scripts\open-webui.exe` exists after installation.

## 4. Ollama Service
- If Ollama is already running as a service, you do NOT need to start it again.
- You can comment out or remove the Ollama start line in `scripts\run_system.bat`:
  ```
  REM start "Ollama" cmd /k "ollama serve"
  ```

## 5. Running Services
- After completing the above, you can run the router and context server in the same window or separate windows, but always ensure `.venv` is activated.
- To run both in the same window:
  ```
  .venv\Scripts\activate
  start /B python src\llm\memory_aware_router.py
  python src\integration\cline_server.py
  ```

## 6. Troubleshooting
- If you see `ModuleNotFoundError`, double-check that the correct environment is activated and dependencies are installed.
- If `open-webui.exe` is not found, ensure you installed Open-WebUI in `.webui-venv` and that you are using the correct version.

## 7. Updating the Launcher Script
- After confirming everything works, you can update `scripts\run_system.bat` to skip starting Ollama if it's already running, and to check for the presence of `open-webui.exe`.

---
This checklist will help you resolve the current issues and get all services running smoothly.
