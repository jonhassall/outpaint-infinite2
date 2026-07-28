from __future__ import annotations

import time

import gradio as gr
from PIL import Image

from .engine import OutpaintEngine
from .jobs import JobManager, TERMINAL_STATES
from .outpaint_helpers import Direction


def build_ui(engine: OutpaintEngine, jobs: JobManager) -> gr.Blocks:
    directions = [direction.value for direction in Direction]

    def one_shot(
        image: Image.Image,
        prompt: str,
        direction: str,
        expand_pixels: int,
        steps: int,
        seed: int,
        randomize_seed: bool,
    ):
        if image is None:
            raise gr.Error("Upload an image first")
        result = engine.outpaint(
            image,
            prompt,
            direction=direction,
            expand_pixels=int(expand_pixels),
            steps=int(steps),
            seed=-1 if randomize_seed else int(seed),
        )
        details = (
            f"Direction: **{result.direction.value}**  \n"
            f"Seed: **{result.seed}**  \n"
            f"Canvas: **{result.canvas_size[0]}×{result.canvas_size[1]}**  \n"
            f"Time: **{result.elapsed_seconds:.2f}s**"
        )
        return result.image, result.seed, details

    def continuous(
        image: Image.Image,
        prompt: str,
        selected_directions: list[str],
        expand_pixels: int,
        steps: int,
        max_steps: int,
        delay_seconds: float,
        seed: int,
        randomize_seed: bool,
    ):
        if image is None:
            raise gr.Error("Upload an image first")
        parsed = [Direction(value) for value in selected_directions]
        job = jobs.create(
            image,
            prompt=prompt,
            directions=parsed,
            expand_pixels=int(expand_pixels),
            steps=int(steps),
            max_steps=int(max_steps),
            delay_seconds=float(delay_seconds),
            randomize_seed=bool(randomize_seed),
            seed=int(seed),
        )

        last_version = -1
        while True:
            snapshot = jobs.wait_for_update(job.id, last_version, timeout=1.0)
            if snapshot is None:
                continue
            last_version = snapshot["version"]
            latest = None
            if job.latest_path and job.latest_path.exists():
                with Image.open(job.latest_path) as opened:
                    latest = opened.convert("RGB").copy()
            status = (
                f"Job: `{job.id}`  \n"
                f"Status: **{snapshot['status']}**  \n"
                f"Step: **{snapshot['current_step']}**"
            )
            if snapshot.get("latest_direction"):
                status += f"  \nDirection: **{snapshot['latest_direction']}**"
            if snapshot.get("latest_seed") is not None:
                status += f"  \nSeed: **{snapshot['latest_seed']}**"
            if snapshot.get("latest_canvas_size"):
                width, height = snapshot["latest_canvas_size"]
                status += f"  \nCanvas: **{width}×{height}**"
            if snapshot.get("latest_source_was_resized"):
                status += "  \nPrevious frame was resized to stay within MAX_CANVAS."
            if snapshot.get("error"):
                status += f"  \nError: `{snapshot['error']}`"
            yield latest, status, job.id
            if snapshot["status"] in TERMINAL_STATES:
                return
            time.sleep(0.1)

    def stop(job_id: str):
        if not job_id:
            return "No active job"
        try:
            snapshot = jobs.stop(job_id).snapshot()
            return f"Stop requested for `{job_id}`. Status: **{snapshot['status']}**"
        except KeyError:
            return "Job no longer exists"

    with gr.Blocks(title="Krea 2 Outpaint Server") as demo:
        gr.Markdown(
            "# Krea 2 Outpaint Server\n"
            "Single-image outpainting plus continuous recursive outpainting. "
            "Each completed frame is saved and reused as the next source."
        )

        with gr.Tab("One shot"):
            with gr.Row():
                with gr.Column():
                    one_image = gr.Image(type="pil", label="Source image")
                    one_prompt = gr.Textbox(
                        label="Prompt",
                        lines=3,
                        placeholder="Describe the complete scene, not only the new area",
                    )
                    one_direction = gr.Radio(
                        directions, value="right", label="Direction"
                    )
                    one_expand = gr.Slider(
                        64, 640, value=256, step=16, label="Expansion pixels"
                    )
                    one_steps = gr.Slider(4, 16, value=8, step=1, label="Steps")
                    with gr.Row():
                        one_seed = gr.Number(value=42, precision=0, label="Seed")
                        one_random = gr.Checkbox(value=True, label="Random seed")
                    one_run = gr.Button("Outpaint", variant="primary")
                with gr.Column():
                    one_output = gr.Image(label="Result", interactive=False)
                    one_details = gr.Markdown()

            one_run.click(
                one_shot,
                inputs=[
                    one_image,
                    one_prompt,
                    one_direction,
                    one_expand,
                    one_steps,
                    one_seed,
                    one_random,
                ],
                outputs=[one_output, one_seed, one_details],
                api_name="outpaint_ui",
                concurrency_limit=4,
            )

        with gr.Tab("Continuous"):
            gr.Markdown(
                "Choose several directions for random movement. Set max steps to `0` "
                "for an unlimited loop; remember that this can fill the output disk."
            )
            job_state = gr.State("")
            with gr.Row():
                with gr.Column():
                    loop_image = gr.Image(type="pil", label="Starting image")
                    loop_prompt = gr.Textbox(
                        label="Prompt",
                        lines=3,
                        placeholder="Describe the complete scene consistently",
                    )
                    loop_directions = gr.CheckboxGroup(
                        directions,
                        value=directions,
                        label="Random directions",
                    )
                    loop_expand = gr.Slider(
                        64, 640, value=256, step=16, label="Expansion pixels"
                    )
                    loop_steps = gr.Slider(4, 16, value=8, step=1, label="Steps")
                    loop_max = gr.Number(
                        value=20, precision=0, label="Max steps (0 = unlimited)"
                    )
                    loop_delay = gr.Number(
                        value=0, label="Delay between completed frames (seconds)"
                    )
                    with gr.Row():
                        loop_seed = gr.Number(value=42, precision=0, label="Seed")
                        loop_random = gr.Checkbox(value=True, label="Random seed each step")
                    with gr.Row():
                        loop_start = gr.Button("Start", variant="primary")
                        loop_stop = gr.Button("Stop", variant="stop")
                with gr.Column():
                    loop_output = gr.Image(label="Live result", interactive=False)
                    loop_status = gr.Markdown("Idle")

            loop_start.click(
                continuous,
                inputs=[
                    loop_image,
                    loop_prompt,
                    loop_directions,
                    loop_expand,
                    loop_steps,
                    loop_max,
                    loop_delay,
                    loop_seed,
                    loop_random,
                ],
                outputs=[loop_output, loop_status, job_state],
                api_name="continuous_ui",
                concurrency_limit=8,
            )
            loop_stop.click(
                stop,
                inputs=[job_state],
                outputs=[loop_status],
                concurrency_limit=8,
            )

        gr.Markdown(
            "REST API docs are available at `/docs`. Generated files are served under `/outputs`."
        )

    return demo.queue(default_concurrency_limit=8, max_size=32)
