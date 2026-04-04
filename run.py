import sys

from llmrouter.app import main


if __name__ == "__main__":
    if "--tray" not in sys.argv:
        sys.argv.append("--tray")
    main()
