"""Optional native-evaluator traceback timer for a private deployment overlay.

Copy as code/overlay/sitecustomize.py. Disabled unless explicitly configured;
only the simulator evaluator entrypoint is instrumented, never the Agent/tools.
"""
import faulthandler
import os
import sys

interval = float(os.environ.get('Q0_SIM_TRACEBACK_INTERVAL_S', '0'))
if interval > 0 and 'eval_client/main.py' in sys.argv[0].replace('\\', '/'):
    faulthandler.enable()
    faulthandler.dump_traceback_later(interval, repeat=True)
