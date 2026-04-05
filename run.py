import sys

from llmrouter.app import main


if __name__ == "__main__":
    # --tray ist immer Standard
    sys.argv.append("--tray")
    main()
