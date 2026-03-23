import runpod
import traceback
import sys
import os
import base64
import tempfile
import requests
import numpy as np

sys.path.insert(0, "/app/ditto-talkinghead")
os.chdir("/app/ditto-talkinghead")

import boto3

SDK = None
run_fn = None
INIT_ERROR = None


def load_model():
    global SDK, run_fn
    from stream_pipeline_offline import StreamSDK
    from inference import run

    SDK = StreamSDK(
        "/app/checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl",
        "/app/checkpoints/ditto_pytorch",
    )
    run_fn = run


try:
    load_model()
except Exception:
    INIT_ERROR = traceback.format_exc()


def download_file(url, suffix):
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=65536):
            tmp.write(chunk)
    tmp.close()
    return tmp.name


def decode_base64_to_file(data, suffix):
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(base64.b64decode(data))
    tmp.close()
    return tmp.name


def build_setup_kwargs(inp):
    sk = {}

    emo = inp.get("emotion", 4)
    if isinstance(emo, list) and emo and isinstance(emo[0], list):
        if emo[0] and isinstance(emo[0][0], float):
            emo = np.array(emo, dtype=np.float32)
    sk["emo"] = emo

    sk["crop_scale"] = float(inp.get("crop_scale", 2.3))
    sk["sampling_timesteps"] = int(inp.get("sampling_timesteps", 50))

    if "crop_vx_ratio" in inp:
        sk["crop_vx_ratio"] = float(inp["crop_vx_ratio"])
    if "crop_vy_ratio" in inp:
        sk["crop_vy_ratio"] = float(inp["crop_vy_ratio"])

    if "use_d_keys" in inp:
        val = inp["use_d_keys"]
        if isinstance(val, list):
            sk["use_d_keys"] = tuple(val)
        else:
            sk["use_d_keys"] = val

    if "drive_eye" in inp:
        sk["drive_eye"] = bool(inp["drive_eye"])
    if "eye_f0_mode" in inp:
        sk["eye_f0_mode"] = bool(inp["eye_f0_mode"])
    if "delta_eye_open_n" in inp:
        sk["delta_eye_open_n"] = inp["delta_eye_open_n"]

    if "overall_ctrl_info" in inp:
        ctrl = dict(inp["overall_ctrl_info"])
        if "delta_exp" in ctrl:
            ctrl["delta_exp"] = np.array(ctrl["delta_exp"], dtype=np.float32).reshape(1, 63)
        sk["overall_ctrl_info"] = ctrl

    if "ctrl_info" in inp:
        raw = inp["ctrl_info"]
        converted = {}
        for fid, ctrl in raw.items():
            ctrl = dict(ctrl)
            if "delta_exp" in ctrl:
                ctrl["delta_exp"] = np.array(ctrl["delta_exp"], dtype=np.float32).reshape(1, 63)
            converted[int(fid)] = ctrl
        sk["ctrl_info"] = converted

    return sk


def upload_to_r2(job_id, file_path):
    s3 = boto3.client("s3",
        endpoint_url=os.environ["BUCKET_ENDPOINT_URL"],
        aws_access_key_id=os.environ["BUCKET_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["BUCKET_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    bucket = os.environ["BUCKET_NAME"]
    key = f"outputs/{job_id}.mp4"
    s3.upload_file(file_path, bucket, key)
    return s3.generate_presigned_url("get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=3600,
    )


def handler(job):
    if INIT_ERROR:
        return {"error": f"Model failed to load:\n{INIT_ERROR}"}

    inp = job["input"]

    audio_url = inp.get("audio_url")
    image_b64 = inp.get("source_image_base64")
    if not audio_url:
        return {"error": "audio_url is required"}
    if not image_b64:
        return {"error": "source_image_base64 is required"}

    audio_path = None
    image_path = None
    out_path = None

    try:
        audio_path = download_file(audio_url, ".wav")
        image_path = decode_base64_to_file(image_b64, ".png")

        tmp_o = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_o.close()
        out_path = tmp_o.name

        more_kwargs = {
            "setup_kwargs": build_setup_kwargs(inp),
            "run_kwargs": {},
        }

        run_fn(SDK, audio_path, image_path, out_path, more_kwargs)

        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            return {"error": "Inference produced no output"}

        video_url = upload_to_r2(job["id"], out_path)

        return {"video_url": video_url, "format": "mp4"}

    except Exception:
        return {"error": traceback.format_exc()}

    finally:
        for p in [audio_path, image_path, out_path]:
            if p and os.path.exists(p):
                os.unlink(p)
        tmp_mp4 = out_path + ".tmp.mp4" if out_path else None
        if tmp_mp4 and os.path.exists(tmp_mp4):
            os.unlink(tmp_mp4)


runpod.serverless.start({"handler": handler})
