"""Allow ``python -m orchestrator <args>`` as an alias for orchestrator.main."""
from orchestrator.main import main
import sys

sys.exit(main())
