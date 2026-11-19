"""Optional native-evaluator traceback timer for a private deployment overlay.

Copy as code/overlay/sitecustomize.py. Disabled unless explicitly configured;
only the simulator evaluator entrypoint is instrumented, never the Agent/tools.
"""
import faulthandler
import os
import sys
import json
import threading

interval = float(os.environ.get('Q0_SIM_TRACEBACK_INTERVAL_S', '0'))
if interval > 0 and 'eval_client/main.py' in sys.argv[0].replace('\\', '/'):
    faulthandler.enable()
    faulthandler.dump_traceback_later(interval, repeat=True)

    def report_settings():
        settings_module = sys.modules.get('carb.settings')
        if settings_module is None:
            return
        settings = settings_module.get_settings()
        keys = ('/app/userConfigPath', '/app/asyncRendering', '/app/asyncRenderingLowLatency',
                '/physics/physxDispatcher', '/persistent/physics/numThreads',
                '/physics/cooking/ujitsoCollisionCooking', '/physics/cooking/ujitsoRemoteCacheEnabled',
                '/persistent/physics/useLocalMeshCache', '/persistent/physics/useActiveCudaContext',
                '/plugins/carb.tasking.plugin/threadCount', '/plugins/omni.tbb.globalcontrol/maxThreadCount')
        print('Q0_SIM_SETTINGS '+json.dumps({key:settings.get(key) for key in keys}),
              file=sys.stderr, flush=True)

    timer = threading.Timer(min(60., interval/2), report_settings)
    timer.daemon = True
    timer.start()
