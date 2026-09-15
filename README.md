# SPLATROOM RunPod Serverless Worker

GPU worker for the SPLATROOM real-estate 3D Gaussian Splatting pipeline.

## Pipeline

`phone video -> download -> ns-process-data / COLMAP -> Splatfacto / gsplat -> Gaussian .ply -> Supabase Storage -> published tour`

The worker never receives a Supabase service-role key. It only receives:

- a temporary signed URL for the source video;
- an opaque one-time callback token for one processing job;
- a callback URL;
- the expected output Storage path.

After training, it requests a fresh signed upload URL from the callback. This avoids an upload URL expiring during long reconstructions.

## RunPod Serverless

Use this repository as the worker source and build the included `Dockerfile`.

Recommended first endpoint configuration:

- Endpoint type: Queue / Serverless
- GPU: A40 48 GB (good cost/VRAM balance)
- GPU count: 1
- Active workers: 0
- Max workers: 1 while testing
- Idle timeout: low/default
- Container disk: 40-60 GB minimum
- Request timeout: enough for a full reconstruction (for example several hours)

The Docker image uses the official Nerfstudio image, which includes Nerfstudio, gsplat, FFmpeg and a CUDA-enabled COLMAP build.

## Input contract

RunPod receives a normal queue job:

```json
{
  "input": {
    "job_id": "uuid",
    "project_id": "uuid",
    "input_url": "temporary https URL",
    "original_name": "house.mp4",
    "output_path": "user/project/scene-job.ply",
    "callback_url": "https://PROJECT.supabase.co/functions/v1/worker-callback",
    "callback_token": "one-time secret",
    "num_frames_target": 300,
    "max_num_iterations": 30000
  }
}
```

These values are produced by SPLATROOM's `dispatch-runpod` Supabase Edge Function. Do not create them manually in production.

## Local Docker preflight

With NVIDIA Container Toolkit installed:

```bash
docker build -t splatroom-worker .
docker run --rm --gpus all splatroom-worker python /app/preflight.py
```

## Cost controls

For testing, keep Active Workers at `0`. This allows RunPod to scale to zero when no house is being processed. Set Max Workers to `1` until the pipeline is proven with real captures.

## Capture quality matters

For a house scan, move slowly, keep strong overlap between views, avoid motion blur, and cover doorways/room transitions from multiple angles. Mirrors, moving people, large blank white walls and rapidly changing exposure can reduce reconstruction quality.
