"""Allow ``python -m orchestrator <args>`` as an alias for orchestrator.main."""
import sys

from orchestrator.main import main

sys.exit(main())
