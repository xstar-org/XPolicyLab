"""Qualify the allocated GPU, Isaac Sim physics and RGB without using any model API."""
import argparse
import json
import os
from pathlib import Path
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    app = None
    try:
        import numpy as np
        import torch
        assert torch.cuda.device_count() == 1
        assert int(torch.arange(1024, device='cuda').sum()) == 523776
        from isaacsim import SimulationApp
        app = SimulationApp(dict(headless=True, renderer='PathTracing', width=320,
                                 height=240, active_gpu=0, multi_gpu=False))
        import isaaclab
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
        import omni.replicator.core as rep
        world = World(stage_units_in_meters=1., physics_dt=1/60, rendering_dt=1/60,
                      backend='numpy', device='cpu')
        world.scene.add(FixedCuboid(prim_path='/World/ground', name='ground',
            position=np.array([0., 0., -.05]), scale=np.array([10., 10., .1])))
        box = world.scene.add(DynamicCuboid(prim_path='/World/box', name='box',
            position=np.array([0., 0., 1.]), scale=np.array([.2, .2, .2]), mass=.1))
        world.reset()
        before = float(box.get_world_pose()[0][2])
        for _ in range(20):
            world.step(render=True)
        after = float(box.get_world_pose()[0][2])
        assert np.isfinite(after) and after < before - .01
        rep.create.light(light_type='Distant', rotation=(315, 0, 0), intensity=3000)
        camera = rep.create.camera(position=(2, 2, 2), look_at=(0, 0, .25))
        product = rep.create.render_product(camera, (320, 240))
        annotator = rep.AnnotatorRegistry.get_annotator('rgb')
        annotator.attach(product)
        for _ in range(8):
            rep.orchestrator.step(rt_subframes=2)
        rgb = np.asarray(annotator.get_data())
        assert rgb.shape[:2] == (240, 320) and float(rgb[..., :3].std()) > 1
        from PIL import Image
        Image.fromarray(rgb).save(args.output/'rgb.png')
        (args.output/'VERIFIED.json').write_text(json.dumps(dict(
            job_id=os.environ['SLURM_JOB_ID'], gpu=torch.cuda.get_device_name(0),
            physics_z_before=before, physics_z_after=after, rgb_shape=list(rgb.shape),
            isaaclab_import=True,
            scope='Infrastructure smoke only; not a push_T episode or model success rate'), indent=2))
        annotator.detach(product)
    except BaseException:
        (args.output/'ERROR.txt').write_text(traceback.format_exc())
        raise
    finally:
        if app is not None:
            app.close()


if __name__ == '__main__':
    main()
